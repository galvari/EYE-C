import argparse
import glob
import json
import random
from pathlib import Path

import cv2
import imageio
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd

from gaze360.model import GazeLSTM
from utils import compute_iou, extract_all_faces, pointsize_to_pointpoint


def gaze_y_other_person(e0, e1, g0):
    # point1 is the origin of the arrow
    # point2 is a second point on the arrow
    point1, point2 = e0, e0 + g0[:2] * [-1, -1]

    # find m: slope
    m = (point1[1] - point2[1]) / (point1[0] - point2[0])
    # find b: the intercept
    b = (point1[0] * point2[1] - point2[0] * point1[1]) / (point1[0] - point2[0])

    # e1 contains eyes coordinates of the other person
    x = e1[0]
    y = m * x + b

    # if (e0[0], point2[0]) -> e1[0] then 0 is looking towards 1
    # elif e[1] - (e0[0], point2[0]) -> 0 is not looking towards 2

    x0 = e0[0]
    x0_gaze = point2[0]
    x1 = e1[0]

    if (x0 < x0_gaze and x0_gaze < x1) or (x1 < x0_gaze and x0_gaze < x0):
        # 0 is looking towards 1
        looking_towards = True
    else:
        looking_towards = False

    return y, looking_towards


def render_frame(image, dists, coordinations):
    image = image.copy()

    h, w, _ = image.shape

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    line_type = 2

    for i, (dist, coordination) in enumerate(zip(dists, coordinations)):
        bottom_left_corner_of_text = (10 + 75 * i, h - 100)

        if np.isnan(dist):
            dist = "nan"
            font_color = (255, 0, 0)  # red
        elif dist > h or dist < 0:
            # if dist is < 0 or greater than height, just show `out`
            dist = "out"
            font_color = (255, 0, 0)  # red
        elif coordination:
            dist = f"{dist:.0f}"
            font_color = (0, 125, 0)  # green
        else:
            dist = f"{dist:.0f}"
            font_color = (255, 200, 0)  # yellow/orange

        cv2.putText(
            image,
            dist,
            bottom_left_corner_of_text,
            font,
            font_scale,
            font_color,
            line_type,
        )

    return image


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize video with openpose heads.")

    parser.add_argument("input_coords", type=str, help="Input coords file")

    parser.add_argument("--input_video", type=str, help="Input video")
    parser.add_argument(
        "--output_folder", type=str, help="Where to store the output video"
    )

    args = parser.parse_args()

    # if only one is not None
    if (args.input_video is None) ^ (args.output_folder is None):
        parser.error("--input_video and --output_folder must be given together")

    return args


def main():
    args = parse_args()

    coords_file = Path(args.input_coords)
    output_json_name = coords_file.parent / f"{coords_file.stem}_coordination.json"

    coords_df = pd.read_json(coords_file)

    y_gazes = []

    # TODO first compute dists and applying filtering

    # gira su persone identificate in coords (person e' una persona)
    print("Finding dists...")
    for i, person in tqdm(coords_df.iterrows(), total=len(coords_df)):
        coordination = False

        # dammi le righe di altre persone (identita' diversa) nello stesso frame
        other_people = coords_df[
            (coords_df.frame == person.frame) & (coords_df.id_t != person.id_t)
        ]

        e0, g0, b0 = map(np.array, person[["eyes", "gaze", "bbox"]])

        dist_min = np.inf
        y_gaze_min = np.inf

        dists_towards_others = []
        dists_towards_me = []

        for _, op in other_people.iterrows():
            e1, g1, b1 = map(np.array, op[["eyes", "gaze", "bbox"]])

            # calcolo distanza da dove interseca vettore g0 la x di e1: y di incrocio - y di e1
            y_gaze_to_other, looking_towards_other = gaze_y_other_person(e0, e1, g0)
            y_gaze_to_me, lookingwards_to_me = gaze_y_other_person(e1, e0, g1)

            if not looking_towards_other:
                dist_towards_other = np.inf
            else:
                dist_towards_other = np.abs(y_gaze_to_other - e1[1])

            if not lookingwards_to_me:
                dist_towards_me = np.inf
            else:
                dist_towards_me = np.abs(y_gaze_to_me - e0[1])

            dists_towards_others.append(dist_towards_other)
            dists_towards_me.append(dist_towards_me)

            # first version of the code, still used to populate next dataframe
            dist = np.abs(y_gaze_to_other - e1[1])

            if dist < dist_min and looking_towards_other:
                dist_min = dist
                y_gaze_min = y_gaze_to_other

        # find if person is involved in coordination
        threshold = 1000

        dists_towards_others = np.array(dists_towards_others)
        dists_towards_me = np.array(dists_towards_me)

        if np.any((dists_towards_others < threshold) & (dists_towards_me < threshold)):
            coordination = True
        else:
            coordination = False

        y_gazes.append(
            {
                "frame": person.frame,
                "id_t": person.id_t,
                "y_gaze": y_gaze_min,
                "dist": dist_min,
                "coordination": coordination,
            }
        )

    y_gazes = pd.DataFrame(y_gazes).set_index(["frame", "id_t"])

    num_frames = y_gazes.index.get_level_values(0).max()

    out = []

    # generate `c` columns for dataframe
    c = ["frame", "num_people"]

    for num_person in range(3):
        c += [
            f"eyes{num_person}",
            f"gaze{num_person}",
            f"bbox{num_person}",
            f"y_gaze{num_person}",
            f"dist{num_person}",
            f"coordination{num_person}",
        ]

    c += ["coordination"]

    print("Generate output json...")
    for frame in tqdm(range(num_frames)):

        # create the frame row
        f = {k: np.nan for k in c}
        f["frame"] = frame

        # find all people in this frame
        people = coords_df[coords_df.frame == frame]
        f["num_people"] = len(people)

        coordination = False

        # if there are 1 to 3 people
        if len(people) > 1 and len(people) <= 3:
            # extract gazes of ALL people in this frame
            y_gazes_frame = y_gazes.loc[frame]

            # iterate on all people in frame
            for i, (_, person) in enumerate(people.iterrows()):
                f[f"eyes{i}"] = person.eyes
                f[f"gaze{i}"] = person.gaze
                f[f"bbox{i}"] = person.bbox

                f[f"y_gaze{i}"] = y_gazes_frame.loc[person.id_t].y_gaze
                f[f"dist{i}"] = y_gazes_frame.loc[person.id_t].dist
                f[f"coordination{i}"] = y_gazes_frame.loc[person.id_t].coordination

                # update coordination variable if coordination was found with this
                # person
                coordination = (
                    coordination or y_gazes_frame.loc[person.id_t].coordination
                )

            f["coordination"] = coordination

        out.append(f)

    out = pd.DataFrame(out)

    out.to_json(output_json_name)

    if args.input_video is not None:
        video_name = args.input_video
        output_video_name = (
            Path(args.output_folder) / f"{Path(video_name).stem}_coordination.mp4"
        )
        video_stream = imageio.get_reader(video_name)
        fps = video_stream.get_meta_data()["fps"]
        out_video = imageio.get_writer(output_video_name, fps=fps)

        print("Writing video...")
        for i, frame in enumerate(
            tqdm(video_stream, total=video_stream.count_frames())
        ):
            dists = out.loc[
                out.frame == i, ["dist0", "dist1", "dist2"]
            ].values.flatten()
            coordinations = out.loc[
                out.frame == i, ["coordination0", "coordination1", "coordination2"]
            ].values.flatten()

            frame = render_frame(frame, dists, coordinations)
            out_video.append_data(frame)

        video_stream.close()
        out_video.close()


if __name__ == "__main__":
    main()

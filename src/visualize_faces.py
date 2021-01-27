import argparse
import glob
import json
from pathlib import Path

import cv2
import imageio
import numpy as np
from tqdm import tqdm

from src.utils import extract_all_faces


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize video with openpose heads.")

    parser.add_argument(
        "input_folder",
        type=str,
        help="Folder where openpose keypoints files are stored",
    )
    parser.add_argument("input_video", type=str, help="Input video")
    parser.add_argument(
        "output_video", type=str, help="Where to store the output video"
    )

    # read and parse command line arguments
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    input_folder = Path(args.input_folder)
    video_name = args.input_video
    output_video_name = Path(args.output_video) / f"{Path(video_name).stem}_heads.mp4"

    # find and sort all json files
    keypoint_files = sorted(glob.glob(str(input_folder / "*.json")))
    video_stream = imageio.get_reader(video_name)

    fps = video_stream.get_meta_data()["fps"]
    out_video = imageio.get_writer(output_video_name, fps=fps)

    for i, json_file in enumerate(tqdm(keypoint_files)):
        frame = video_stream.get_next_data()
        with open(json_file) as f:
            j = json.load(f)

        people = j["people"]
        faces_pts, heads_bbox = extract_all_faces(people)

        for face, head in zip(faces_pts, heads_bbox):
            cv2.rectangle(
                frame,
                tuple(head[:2].round().astype(np.int)),
                tuple((head[:2] + head[2:]).round().astype(np.int)),
                (255, 0, 0),
                3,
            )

        frame = frame.astype(np.uint8)
        out_video.append_data(frame)
    out_video.close()


if __name__ == "__main__":
    main()

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

from gaze360.model import GazeLSTM
from utils import compute_iou, extract_all_faces, pointsize_to_pointpoint


def spherical2cartesial(x):
    output = torch.zeros(x.size(0), 3)
    output[:, 2] = -torch.cos(x[:, 1]) * torch.cos(x[:, 0])
    output[:, 0] = torch.cos(x[:, 1]) * torch.sin(x[:, 0])
    output[:, 1] = torch.sin(x[:, 1])
    return output


def find_id(bbox, id_dict):
    id_final = None
    max_iou = 0.5
    for k in id_dict.keys():
        if compute_iou(bbox, id_dict[k][0]) > max_iou:
            id_final = k
            max_iou = compute_iou(bbox, id_dict[k][0])
    return id_final


def render_frame(image, eyes, gaze, bbox):
    image = image.copy()
    scale = 150 * np.array((-1, -1))
    end_point = eyes + gaze[:2] * scale
    cv2.rectangle(
        image,
        tuple(bbox[:2].round().astype(np.int)),
        tuple((bbox[2:]).round().astype(np.int)),
        (255, 0, 0),
        3,
    )
    cv2.arrowedLine(
        image,
        tuple(i for i in eyes.astype(np.int)),
        tuple(i for i in end_point.astype(np.int)),
        (255, 255, 0),
        thickness=4,
    )
    return image


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize video with openpose heads.")

    parser.add_argument(
        "input_folder",
        type=str,
        help="Folder where openpose keypoints files are stored",
    )
    parser.add_argument("input_video", type=str, help="Input video")
    parser.add_argument(
        "output_video", type=str, help="Folder where to store the output video"
    )
    parser.add_argument("output_json", type=str, help="Folder where to store the json")
    parser.add_argument("--use_cuda", action="store_true", help="Whether to use GPU")
    parser.add_argument(
        "--model_weights",
        type=str,
        help="Checkpoint file of gaze360 model",
        default="models/gaze360_model.pth.tar",
    )

    # read and parse command line arguments
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    input_folder = Path(args.input_folder)
    video_name = args.input_video
    output_video_name = Path(args.output_video) / f"{Path(video_name).stem}_gaze.mp4"
    output_json_name = Path(args.output_json) / f"{Path(video_name).stem}_coords.json"

    # find and sort all json files
    keypoint_files = sorted(glob.glob(str(input_folder / "*.json")))

    final_results = {}
    for i, json_file in enumerate(tqdm(keypoint_files)):
        with open(json_file) as f:
            j = json.load(f)

        people = j["people"]
        _, heads_bbox = extract_all_faces(people)

        if len(heads_bbox) > 0:
            final_results[i] = heads_bbox

    # le heads che calcolavo con una delle prime funzioni le usavo per disegnare
    # sui frame e avevo bisogno che fossero 4 numeri messi cosí: (x, y, h, w) mentre
    # il codice per ottenere gli id delle persone e poi le sequenze funziona con
    # (x1, y1, x2, y2) quindi convertiamo le bbox contenute in final result

    for frame, heads in final_results.items():
        for i, h in enumerate(heads):
            heads[i] = pointsize_to_pointpoint(h)

    id_num = 0
    tracking_id = dict()
    identity_last = dict()
    frames_with_people = list(final_results.keys())

    frames_with_people.sort()
    for i in frames_with_people:
        speople = final_results[i]
        identity_next = dict()
        for j in range(len(speople)):
            bbox_head = speople[j]
            if bbox_head is None:
                continue

            id_val = find_id(bbox_head, identity_last)

            if id_val is None:
                id_num += 1
                id_val = id_num

            # TODO: Improve eye location
            eyes = [
                (bbox_head[0] + bbox_head[2]) / 2.0,
                (0.65 * bbox_head[1] + 0.35 * bbox_head[3]),
            ]
            identity_next[id_val] = (bbox_head, eyes)
        identity_last = identity_next
        tracking_id[i] = identity_last

        # LOAD MODEL
    device = "cuda" if args.use_cuda else "cpu"

    model = GazeLSTM()
    model = torch.nn.DataParallel(model).to(device)
    model.to(device)
    checkpoint = torch.load(args.model_weights)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    image_normalize = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    video_stream = imageio.get_reader(video_name)
    fps = video_stream.get_meta_data()["fps"]
    out_video = imageio.get_writer(output_video_name, fps=fps)

    color_encoding = []
    for i in range(video_stream.count_frames()):
        color_encoding.append(
            [random.randint(0, 254), random.randint(0, 254), random.randint(0, 254)]
        )

    W = max(int(fps // 8), 1)

    frames = {}
    n_frames_init = 50
    frames_to_keep_in_memory = n_frames_init * 2
    n_frames = video_stream.count_frames()
    n_frames_read = 0
    first_frame_in_dict = 0

    # read only the beginning of the video
    for i in range(n_frames_init):
        frames[i] = video_stream.get_next_data()
        n_frames_read += 1

    # frame shape is H, W, C
    HEIGHT, WIDTH = frames[0].shape[:2]


    coords = []

    print("Excuting gaze360...")
    # for each frame in the video
    for i in tqdm(range(n_frames)):
        image = frames[i].copy()

        # TODO check this
        image = cv2.resize(image, (WIDTH, HEIGHT))
        image = image.astype(float)

        # if there are people identified in this frame
        if i in tracking_id:
            # for each identified person
            for id_t in tracking_id[i].keys():
                coord = {}
                # input_image shape is: (seq_len x ch x W x H)
                input_image = torch.zeros(7, 3, 224, 224)
                count = 0

                # extract frames from the past and the future to create the input
                # sequence for gaze360
                for j in range(i - 3 * W, i + 4 * W, W):
                    # if frame `j` contains person `id_t`
                    if j in tracking_id and id_t in tracking_id[j]:
                        new_im = Image.fromarray(frames[j], "RGB")
                        bbox, eyes = tracking_id[j][id_t]
                    else:
                        new_im = Image.fromarray(frames[i], "RGB")
                        bbox, eyes = tracking_id[i][id_t]

                    # crop the head out of the frame
                    new_im = new_im.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                    input_image[count, :, :, :] = image_normalize(new_im)
                    count = count + 1

                # run the model
                output_gaze, _ = model(
                    input_image.view(1, 7, 3, 224, 224).to(device)
                )
                gaze = spherical2cartesial(output_gaze).detach().numpy()
                gaze = gaze.reshape((-1))

                bbox, eyes = tracking_id[i][id_t]
                bbox = np.asarray(bbox).astype(int)
                eyes = np.asarray(eyes).astype(float)

                image = render_frame(image, eyes, gaze, bbox)
                coord = {
                    "frame": i,
                    "id_t": id_t,
                    "eyes": eyes,
                    "gaze": gaze,
                    "bbox": bbox,
                }
                coords.append(coord)

        # read next frame, until there are frames to read
        if n_frames_read < n_frames:
            frames[n_frames_read] = video_stream.get_next_data()
            n_frames_read += 1

        if n_frames_read - first_frame_in_dict > frames_to_keep_in_memory:
            del frames[first_frame_in_dict]
            first_frame_in_dict += 1

        image = image.astype(np.uint8)
        out_video.append_data(image)
    
    video_stream.close()
    out_video.close()

    coords = pd.DataFrame(coords)
    coords.to_json(output_json_name)


if __name__ == "__main__":
    main()

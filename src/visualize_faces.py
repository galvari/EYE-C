# visualize_faces.py
# Author: Gianpaolo Alvari, Luca Coviello
# Description: This script extracts head bounding boxes from OpenPose keypoint JSON files and renders a video with bounding boxes drawn around detected heads.

import argparse
import glob
import json
from pathlib import Path

import cv2
import imageio
import numpy as np
from tqdm import tqdm

from utils import extract_all_faces

# ---------------------------------------------
# Argument Parser
# ---------------------------------------------

def parse_args():
    """
    Parses command-line arguments for the script.
    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Visualize video with OpenPose head bounding boxes.")

    parser.add_argument(
        "input_folder",
        type=str,
        help="Folder where OpenPose keypoints JSON files are stored.",
    )
    parser.add_argument("input_video", type=str, help="Path to the input video file.")
    parser.add_argument(
        "output_folder",
        type=str,
        help="Folder where the output video with head bounding boxes will be saved.",
    )

    return parser.parse_args()

# ---------------------------------------------
# Main Function
# ---------------------------------------------

def main():
    args = parse_args()

    input_folder = Path(args.input_folder)
    video_name = args.input_video
    output_video_name = Path(args.output_folder) / f"{Path(video_name).stem}_heads.mp4"

    # Find and sort all JSON files in the input folder
    keypoint_files = sorted(glob.glob(str(input_folder / "*.json")))

    # Open the input video and prepare the output video writer
    video_stream = imageio.get_reader(video_name)
    fps = video_stream.get_meta_data()["fps"]
    out_video = imageio.get_writer(output_video_name, fps=fps)

    print("Processing video frames and drawing head bounding boxes...")

    for i, json_file in enumerate(tqdm(keypoint_files)):
        frame = video_stream.get_next_data()
        with open(json_file) as f:
            j = json.load(f)

        # Extract people information from the JSON file
        people = j["people"]
        faces_pts, heads_bbox = extract_all_faces(people)

        # Draw bounding boxes around detected heads
        for face, head in zip(faces_pts, heads_bbox):
            cv2.rectangle(
                frame,
                tuple(head[:2].round().astype(np.int32)),
                tuple((head[:2] + head[2:]).round().astype(np.int32)),
                (255, 0, 0),
                3,
            )

        # Convert frame to uint8 type and write to the output video
        frame = frame.astype(np.uint8)
        out_video.append_data(frame)

    # Close the video streams
    video_stream.close()
    out_video.close()

    print(f"Output video saved to: {output_video_name}")

if __name__ == "__main__":
    main()

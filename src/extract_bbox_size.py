# extract_bbox_size.py
# Author: Gianpaolo Alvari, Luca Coviello
# Description: This script extracts bounding box (bbox) sizes from OpenPose keypoint JSON files and saves the data in a CSV file. The bounding boxes represent detected head regions in each video frame.

import argparse
import glob
import json
from pathlib import Path
import pandas as pd
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
    parser = argparse.ArgumentParser(description="Extract bounding box sizes from OpenPose keypoints files.")

    parser.add_argument(
        "input_folder",
        type=str,
        help="Folder where OpenPose keypoints files are stored",
    )
    parser.add_argument("input_video", type=str, help="Input video file")
    parser.add_argument(
        "output_folder", type=str, help="Folder to store the output CSV file"
    )

    return parser.parse_args()

# ---------------------------------------------
# Main Function
# ---------------------------------------------

def main():
    args = parse_args()

    # Paths for input and output
    input_folder = Path(args.input_folder)
    video_name = args.input_video
    output_csv_name = Path(args.output_folder) / f"{Path(video_name).stem}_heads.csv"

    # Find and sort all JSON files in the input folder
    keypoint_files = sorted(glob.glob(str(input_folder / "*.json")))

    # Initialize an empty list to store head bounding box data
    hh = []

    print("Extracting bounding boxes...")

    # Process each keypoint file
    for i, json_file in enumerate(tqdm(keypoint_files)):
        with open(json_file) as f:
            j = json.load(f)

        people = j["people"]

        # Extract face points and head bounding boxes using utility function
        faces_pts, heads_bbox = extract_all_faces(people)

        # Store bounding box data
        for head in heads_bbox:
            hh.append(head)

    # Convert the bounding box data to a DataFrame and save it as a CSV file
    hh_df = pd.DataFrame(hh, columns=["x", "y", "width", "height"])
    hh_df.to_csv(output_csv_name, index=False)

    print(f"Bounding box sizes saved to: {output_csv_name}")

if __name__ == "__main__":
    main()

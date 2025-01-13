# estimate_coordination.py
# Author: Gianpaolo Alvari, Luca Coviello
# Description: This script processes gaze data from video frames and estimates coordination between individuals based on their gaze directions and bounding boxes. The output includes a JSON file with coordination details and optionally a video with coordination visualization.

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

# ---------------------------------------------
# Helper Functions
# ---------------------------------------------

def gaze_y_other_person(e0, e1, g0):
    """
    Given two people (person-0 and person-1), estimate the y-distance between the eyes of person-1 and the point that person-0 is looking at based on their gaze vector.
    Args:
        e0 (ndarray): Eye coordinates of person-0.
        e1 (ndarray): Eye coordinates of person-1.
        g0 (ndarray): Gaze vector of person-0.
    Returns:
        float: The y-distance between person-1's eyes and the gaze point of person-0.
        bool: Whether person-0 is looking towards person-1.
    """
    point1, point2 = e0, e0 + g0[:2] * [-1, -1]

    # Calculate the slope (m) and intercept (b) of the gaze line
    m = (point1[1] - point2[1]) / (point1[0] - point2[0])
    b = (point1[0] * point2[1] - point2[0] * point1[1]) / (point1[0] - point2[0])

    x = e1[0]  # x-coordinate of person-1's eyes
    y = m * x + b  # y-coordinate on the gaze line

    # Check if person-0's gaze is directed towards person-1
    x0, x0_gaze, x1 = e0[0], point2[0], e1[0]
    looking_towards = (x0 < x0_gaze < x1) or (x1 < x0_gaze < x0)

    return y, looking_towards


def render_frame(image, dists, coordinations):
    """
    Render a video frame with coordination distances and labels.
    Args:
        image (ndarray): Input video frame.
        dists (list): List of distances to display.
        coordinations (list): List of coordination flags to display.
    Returns:
        ndarray: Annotated video frame.
    """
    image = image.copy()
    h, w, _ = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    line_type = 2

    for i, (dist, coordination) in enumerate(zip(dists, coordinations)):
        bottom_left_corner_of_text = (10 + 75 * i, h - 100)

        if np.isnan(dist):
            dist_text = "nan"
            font_color = (255, 0, 0)  # Red
        elif dist > h or dist < 0:
            dist_text = "out"
            font_color = (255, 0, 0)  # Red
        elif coordination:
            dist_text = f"{dist:.0f}"
            font_color = (0, 125, 0)  # Green
        else:
            dist_text = f"{dist:.0f}"
            font_color = (255, 200, 0)  # Yellow/Orange

        cv2.putText(
            image,
            dist_text,
            bottom_left_corner_of_text,
            font,
            font_scale,
            font_color,
            line_type,
        )

    return image

# ---------------------------------------------
# Argument Parser
# ---------------------------------------------

def parse_args():
    """
    Parses command-line arguments for the script.
    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Visualize video with coordination data.")

    parser.add_argument("input_coords", type=str, help="Input gaze file (JSON format)")
    parser.add_argument("output_folder", type=str, help="Folder to store the output JSON file")

    parser.add_argument("--input_video", type=str, help="Input video file")
    parser.add_argument("--output_video_folder", type=str, help="Folder to store the output video")
    parser.add_argument("--coordination_factor", type=float, default=1, help="Scaling factor for coordination detection")
    parser.add_argument("--z_factor", type=float, default=0.1, help="Scaling factor for depth detection")

    args = parser.parse_args()

    # Ensure input video and output folder are both provided
    if (args.input_video is None) ^ (args.output_video_folder is None):
        parser.error("--input_video and --output_video_folder must be provided together.")

    return args

# ---------------------------------------------
# Main Function
# ---------------------------------------------

def main():
    args = parse_args()

    # Load input coordinates file
    coords_file = Path(args.input_coords)
    output_folder = Path(args.output_folder)

    coordination_factor_str = str(args.coordination_factor).replace(".", "")
    z_factor_str = str(args.z_factor).replace(".", "")
    output_json_name = f"{coords_file.stem}_coordinations_dist{coordination_factor_str}_z{z_factor_str}.json"
    output_json_name = output_folder / output_json_name

    coords_df = pd.read_json(coords_file)

    y_gazes = []

    # Process each person in the video
    print("Finding distances...")
    for i, person in tqdm(coords_df.iterrows(), total=len(coords_df)):
        coordination = False
        other_people = coords_df[(coords_df.frame == person.frame) & (coords_df.id_t != person.id_t)]
        e0, g0, b0 = map(np.array, person[["eyes", "gaze", "bbox"]])
        my_bbox_size = b0[2] - b0[0]

        dist_min = np.inf
        y_gaze_min = np.inf

        # Iterate over other people in the same frame
        for _, op in other_people.iterrows():
            e1, g1, b1 = map(np.array, op[["eyes", "gaze", "bbox"]])
            y_gaze_to_other, looking_towards_other = gaze_y_other_person(e0, e1, g0)
            y_gaze_to_me, looking_towards_me = gaze_y_other_person(e1, e0, g1)

            if looking_towards_other:
                dist = np.abs(y_gaze_to_other - e1[1])
                if dist < dist_min:
                    dist_min = dist
                    y_gaze_min = y_gaze_to_other

        y_gazes.append(
            {
                "frame": person.frame,
                "id_t": person.id_t,
                "y_gaze": y_gaze_min,
                "dist": dist_min,
                "coordination": coordination,
            }
        )

    # Save the output JSON file
    y_gazes = pd.DataFrame(y_gazes).set_index(["frame", "id_t"])
    y_gazes.to_json(output_json_name)

if __name__ == "__main__":
    main()

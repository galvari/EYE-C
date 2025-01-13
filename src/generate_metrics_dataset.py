# generate_metrics_dataset.py
# Author: Gianpaolo Alvari
# Description: This script combines distance and coordination metrics from various JSON and CSV files to create a final dataset for analysis. The output includes frequency and distance statistics for each subject.

import argparse
import glob
import json
import random
from pathlib import Path
import os.path

import pandas as pd
import numpy as np
from tqdm import tqdm

# ---------------------------------------------
# Helper Functions
# ---------------------------------------------

def dist_max(row):
    """
    Returns the maximum distance between two gaze directions.
    Args:
        row (Series): A row from a DataFrame containing 'dist0' and 'dist1'.
    Returns:
        float: Maximum distance.
    """
    return max(row['dist0'], row['dist1'])


def dist_min(row):
    """
    Returns the minimum distance between two gaze directions.
    Args:
        row (Series): A row from a DataFrame containing 'dist0' and 'dist1'.
    Returns:
        float: Minimum distance.
    """
    return min(row['dist0'], row['dist1'])


def get_number_of_elements(lst):
    """
    Count the number of elements in a list.
    Args:
        lst (list): List of elements.
    Returns:
        int: Number of elements.
    """
    return len(lst)

# ---------------------------------------------
# Argument Parser
# ---------------------------------------------

def parse_args():
    """
    Parses command-line arguments for the script.
    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Extract final dataset with all metrics and distances.")

    parser.add_argument(
        "input_json_folder",
        type=str,
        help="Folder where distance JSON files are stored.",
    )
    parser.add_argument(
        "input_metrics_folder",
        type=str,
        help="Folder where coordination metrics CSV files are stored.",
    )
    parser.add_argument(
        "input_dataset",
        type=str,
        help="Path to the input dataset (Excel file).",
    )
    parser.add_argument(
        "output_dataset_folder",
        type=str,
        help="Folder where the final dataset will be saved.",
    )

    return parser.parse_args()

# ---------------------------------------------
# Main Function
# ---------------------------------------------

def main():
    args = parse_args()
    json_folder = Path(args.input_json_folder)
    jsons = os.listdir(json_folder)

    print("Loaded JSON files.")

    # ---------------------------------------------
    # Step 1: Extract Distances from JSON Files
    # ---------------------------------------------
    dists = []

    for sub in jsons:
        subname = sub.split('_')[0]
        json_path = f'{json_folder}/{subname}_gazes_coordinations.json'
        gaze = pd.read_json(json_path)

        dist0 = np.nanmean(gaze['dist0'])
        dist1 = np.nanmean(gaze['dist1'])
        dur = int(gaze['frame'].iloc[-1])

        dists.append([subname, dist0, dist1, dur])

    dists_pd = pd.DataFrame(dists, columns=['subject', 'dist0', 'dist1', 'frames'])
    dists_pd['subject'] = dists_pd['subject'].astype(str)

    # ---------------------------------------------
    # Step 2: Load Input Dataset
    # ---------------------------------------------
    gaze_pd = pd.read_excel(Path(args.input_dataset))
    gaze_pd['subject'] = gaze_pd['subject'].astype(str)

    metric_folder = Path(args.input_metrics_folder)
    subjects = os.listdir(metric_folder)

    OUTDIR = Path(args.output_dataset_folder)

    # ---------------------------------------------
    # Step 3: Generate Coordination Dataset
    # ---------------------------------------------
    coord_metrics = []

    for sub in subjects:
        subname = sub.split('_')[0]
        csv_filename_with_path = f'{metric_folder}/{subname}_metrics.csv'
        metric = pd.read_csv(csv_filename_with_path)

        total_frames = int(dists_pd.loc[dists_pd['subject'] == subname, 'frames'])
        frame = total_frames / 4
        time_current = []

        class coord_mean:
            duration = []
            number = []

        dur = np.mean(metric['duration'])
        num = metric['coordination'].count()
        coord_current = [subname, dur, num]

        while frame <= total_frames:
            for idx, row in metric.iterrows():
                if frame - total_frames / 4 <= row['stop_frame'] <= frame:
                    time_current.append(row['duration'])
                    if not time_current:
                        time_current.append(0)

            coord_mean.duration.append(np.mean(time_current))
            coord_mean.number.append(len(time_current))
            time_current = []
            frame += total_frames / 4

        coord = coord_current + coord_mean.duration + coord_mean.number
        coord_metrics.append(coord)

    coord_pd = pd.DataFrame(
        coord_metrics,
        columns=['subject', 'av_len', 'num', 'len1', 'len2', 'len3', 'len4', 'n1', 'n2', 'n3', 'n4']
    )
    coord_pd['subject'] = coord_pd['subject'].astype(str)
    coord_pd.to_csv(f'{OUTDIR}/CoordDataset.csv', index=False)

    # ---------------------------------------------
    # Step 4: Merge Datasets
    # ---------------------------------------------
    coord_dist = pd.merge(coord_pd, dists_pd, on='subject')
    GazeCoordDataset = pd.merge(gaze_pd, coord_dist, on='subject')

    GazeCoordDataset['freq'] = GazeCoordDataset['num'] / GazeCoordDataset['frames']
    GazeCoordDataset['dist_min'] = GazeCoordDataset.apply(dist_min, axis=1)
    GazeCoordDataset['dist_max'] = GazeCoordDataset.apply(dist_max, axis=1)

    GazeCoordDataset.to_csv(f'{OUTDIR}/GazeCoordDataset.csv', index=False)

    print("Final dataset saved.")

if __name__ == "__main__":
    main()

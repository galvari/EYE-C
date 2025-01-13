# generate_coordination_dataset.py
# Author: Gianpaolo Alvari
# Description: This script generates a dataset of gaze coordination metrics by processing multiple JSON and CSV files. It calculates various coordination statistics and stores the results in a merged dataset as a CSV file.

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import re
import os.path

# List of parameter configurations to process
params_list = ['dist08_z06', 'dist08_z03', 'dist08_z09']

# ---------------------------------------------
# Helper Functions
# ---------------------------------------------

def dist_max(row):
    """
    Compute the maximum distance mean and its standard deviation from two distance columns.
    Args:
        row (Series): Row of the dataframe containing 'dist0_mean' and 'dist1_mean'.
    Returns:
        tuple: Maximum distance mean and its standard deviation.
    """
    if row['dist1_mean'] > row['dist0_mean']:
        return row['dist1_mean'], row['dist1_std']
    else:
        return row['dist0_mean'], row['dist0_std']


def dist_min(row):
    """
    Compute the minimum distance mean and its standard deviation from two distance columns.
    Args:
        row (Series): Row of the dataframe containing 'dist0_mean' and 'dist1_mean'.
    Returns:
        tuple: Minimum distance mean and its standard deviation.
    """
    if row['dist1_mean'] < row['dist0_mean']:
        return row['dist1_mean'], row['dist1_std']
    else:
        return row['dist0_mean'], row['dist0_std']


def get_number_of_elements(lst):
    """
    Count the number of elements in a list.
    Args:
        lst (list): List of elements.
    Returns:
        int: Number of elements in the list.
    """
    return len(lst)

# ---------------------------------------------
# Main Processing Loop
# ---------------------------------------------

# Process each parameter configuration in the params_list
for params in params_list:
    json_folder = f".../JSON_coordination/{params}"
    jsons = os.listdir(json_folder)
    json_subnames = np.unique([x.split('_')[1] for x in jsons])

    print(f"Processing {params} - Sample size: {len(json_subnames)}")

    metric_folder = f'.../Coord_Metrics/{params}'
    OUTDIR = '.../ADOS_Pietro/Dataset'

    dists = []

    # ---------------------------------------------
    # Extract Distances from JSON Files
    # ---------------------------------------------
    for sub in jsons:
        match = re.search(r'lab\d+', sub)
        subname = match.group() if match else sub.split('_')[0]

        json_path = f'{json_folder}/{sub}'
        gaze = pd.read_json(json_path)

        dist0_mean = gaze.dist0.mean(skipna=True)
        dist0_std = gaze.dist0.std(skipna=True)
        dist1_mean = gaze.dist1.mean(skipna=True)
        dist1_std = gaze.dist1.std(skipna=True)
        dur = int(gaze.frame.iloc[-1])

        dists.append([subname, dist0_mean, dist0_std, dist1_mean, dist1_std, dur])

    # Create a DataFrame from the extracted distances
    dists_pd = pd.DataFrame(dists, columns=['subject', 'dist0_mean', 'dist0_std', 'dist1_mean', 'dist1_std', 'frames'])
    dists_pd.subject = dists_pd.subject.astype(str)

    dists_pd = dists_pd.groupby('subject').agg({
        'dist0_mean': 'mean',
        'dist0_std': 'mean',
        'dist1_mean': 'mean',
        'dist1_std': 'mean',
        'frames': 'sum'
    }).reset_index()

    # ---------------------------------------------
    # Merge Metrics Files by Subject
    # ---------------------------------------------
    file_groups = {}
    for filename in os.listdir(metric_folder):
        if filename.endswith("_metrics.csv") and 'merged' not in filename:
            subjectname = filename.split('_')[0]
            file_groups.setdefault(subjectname, []).append(filename)

    for subjectname, filenames in file_groups.items():
        print(f"Merging metrics for {subjectname}...")

        merged_df = pd.DataFrame()
        total_video_length = 0

        for file in filenames:
            print(f"Processing {file}...")
            df = pd.read_csv(os.path.join(metric_folder, file))

            if 'video_length' in df.columns and not df['video_length'].isnull().all():
                total_video_length += df['video_length'].iloc[0]
            else:
                print(f"Warning: 'video_length' column not found in {file}")

            merged_df = pd.concat([merged_df, df], ignore_index=True)

        merged_df['video_length'] = total_video_length
        merged_df.to_csv(os.path.join(metric_folder, f"{subjectname}_merged_metrics.csv"), index=False)

    # ---------------------------------------------
    # Generate Coordination Dataset
    # ---------------------------------------------
    coord_metrics = []
    fps = 25

    for sub in os.listdir(metric_folder):
        if 'merged' in sub:
            subname = sub.split('_')[0]
            metric = pd.read_csv(f'{metric_folder}/{subname}_merged_metrics.csv')

            total_len = int(dists_pd.frames[dists_pd.subject == subname].iloc[0]) / fps
            video_len = metric['video_length'].iloc[0] / fps
            frame = total_len / 4
            time_current = []

            class coord_mean:
                duration = []
                number = []

            dur = np.mean(metric.duration)
            num = metric['coordination'].count()
            coord_current = [subname, dur, num, video_len, frame]

            start_stop_frames = metric[['start_frame', 'stop_frame']].sort_values(by='start_frame').values
            time_between_episodes = np.diff(start_stop_frames.flatten())[1::2] / fps

            avg_time_between = np.mean(time_between_episodes) if len(time_between_episodes) > 0 else 0
            std_time_between = np.std(time_between_episodes) if len(time_between_episodes) > 0 else 0

            n_q1, n_q2, n_q3, n_q4 = 0, 0, 0, 0

            while frame <= total_len:
                for idx, row in metric.iterrows():
                    if frame * fps >= row['stop_frame'] >= (frame - total_len / 4) * fps:
                        time_current.append(row['duration'])

                        if frame <= total_len / 4:
                            n_q1 += 1
                        elif frame <= total_len / 2:
                            n_q2 += 1
                        elif frame <= 3 * total_len / 4:
                            n_q3 += 1
                        else:
                            n_q4 += 1

                coord_mean.duration.append(np.mean(time_current))
                coord_mean.number.append(len(time_current))
                time_current = []
                frame += total_len / 4

            coord = coord_current + coord_mean.duration + coord_mean.number
            coord.extend([avg_time_between, std_time_between])
            coord_metrics.append(coord)

    coord_pd = pd.DataFrame(coord_metrics, columns=[
        'subject', 'av_len', 'num', 'video_len', 'quarter_len', 'len_q1', 'len_q2', 'len_q3', 'len_q4',
        'n_q1', 'n_q2', 'n_q3', 'n_q4', 'avg_time_between', 'std_time_between'
    ])

    coord_pd['freq'] = coord_pd['num'] / coord_pd['video_len']
    coord_pd['freq_q1'] = coord_pd['n_q1'] / coord_pd['quarter_len']
    coord_pd['freq_q2'] = coord_pd['n_q2'] / coord_pd['quarter_len']
    coord_pd['freq_q3'] = coord_pd['n_q3'] / coord_pd['quarter_len']
    coord_pd['freq_q4'] = coord_pd['n_q4'] / coord_pd['quarter_len']

    GazeCoordDataset = pd.merge(coord_pd, dists_pd, on='subject')
    GazeCoordDataset['dist_min_mean'], GazeCoordDataset['dist_min_std'] = zip(*GazeCoordDataset.apply(dist_min, axis=1))
    GazeCoordDataset['dist_max_mean'], GazeCoordDataset['dist_max_std'] = zip(*GazeCoordDataset.apply(dist_max, axis=1))
    GazeCoordDataset['subject'] = pd.to_numeric(GazeCoordDataset['subject'].str.extract(r'(\d+)')[0], errors='coerce')

    GazeCoordDataset.to_csv(f'{OUTDIR}/GazeCoordDataset_{params}.csv', index=False)

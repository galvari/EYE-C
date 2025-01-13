# generate_metrics_files.py
# Author: Gianpaolo Alvari
# Description: This script processes gaze coordination JSON files to extract metrics such as coordination episodes and their durations. The extracted metrics are saved as CSV files for each subject.

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from scenedetect.frame_timecode import FrameTimecode
import re
import os.path

# ---------------------------------------------
# List of parameter configurations to process
# ---------------------------------------------
params_list = ['dist08_z06', 'dist08_z09', 'dist08_z03']

# ---------------------------------------------
# Helper Functions
# ---------------------------------------------

def get_new_filename(base_path, subname):
    """
    Generate a unique filename for the output CSV to avoid overwriting existing files.
    Args:
        base_path (str): Base directory path where the file will be saved.
        subname (str): Subject name to include in the filename.
    Returns:
        str: Unique filename with a counter suffix if the file already exists.
    """
    counter = 1
    original_filename = f"{base_path}/{subname}_metrics.csv"
    new_filename = original_filename

    while os.path.exists(new_filename):
        new_filename = f"{base_path}/{subname}_{counter}_metrics.csv"
        counter += 1

    return new_filename


def get_indexes(x):
    """
    Compute the duration of a coordination episode in seconds based on frame count.
    Args:
        x (Series): Coordination episode frames.
    Returns:
        float: Duration in seconds.
    """
    duration = len(x) / 25  # Assuming 25 FPS
    return duration


def get_st_sp(x):
    """
    Identify start and stop indexes of coordination episodes from a binary sequence.
    Args:
        x (Series): Binary sequence indicating coordination (1) or no coordination (0).
    Returns:
        tuple: Start and stop indexes of coordination episodes.
    """
    idx_peaks = x > 0
    idx_peaks = idx_peaks.astype(int)
    idx_peaks_diff = np.diff(idx_peaks)

    idx_start = np.where(idx_peaks_diff == 1)[0] + 1
    idx_stop = np.where(idx_peaks_diff == -1)[0] + 1

    return idx_start, idx_stop

# ---------------------------------------------
# Main Processing Loop
# ---------------------------------------------
for params in params_list:
    json_folder = f".../JSON_coordination/{params}"
    jsons = os.listdir(json_folder)

    OUTDIR = f".../Coord_Metrics/{params}"
    os.makedirs(OUTDIR, exist_ok=True)

    json_subnames = np.unique([x.split('_')[1] for x in jsons])
    print(f"Found: {json_subnames}")

    for sub in jsons:
        print(f"Extracting: {sub}")

        # Extract subject name from filename
        match = re.search(r'lab\d+', sub)
        subname = match.group() if match else sub.split('_')[0]

        json_filename_with_path = f"{json_folder}/{sub}"
        gaze = pd.read_json(json_filename_with_path)

        # ---------------------------------------------
        # Preprocess Coordination Data
        # ---------------------------------------------
        gaze['coordination'] = gaze['coordination'].fillna(0)

        new_row = pd.DataFrame({
            'frame': -1, 'num_people': 0, 'eyes0': 0, 'gaze0': 0, 'bbox0': 0, 'y_gaze0': 0, 'dist0': 0,
            'coordination0': 0, 'eyes1': 0, 'gaze1': 0, 'bbox1': 0, 'y_gaze1': 0, 'dist1': 0,
            'coordination1': 0, 'eyes2': 0, 'gaze2': 0, 'bbox2': 0, 'y_gaze2': 0, 'dist2': 0,
            'coordination2': 0, 'coordination': 0
        }, index=[0])
        gaze = pd.concat([new_row, gaze, new_row]).reset_index(drop=True)

        # ---------------------------------------------
        # Fill Small Gaps in Coordination Sequences
        # ---------------------------------------------
        gaze['coord_holes'] = gaze['coordination']

        for i in np.arange(1, len(gaze['coordination']) - 1):
            if gaze['coordination'].iloc[i] == 0:
                if gaze['coordination'].iloc[i - 1] == 1 and gaze['coordination'].iloc[i + 1] == 1:
                    idx_start = i
                    idx_stop = i
                    while idx_stop < len(gaze['coordination']) - 1 and gaze['coordination'].iloc[idx_stop + 1] == 0:
                        idx_stop += 1
                    if (idx_stop - idx_start + 1) < 4:
                        gaze['coord_holes'].iloc[idx_start:idx_stop + 1] = 1

        # ---------------------------------------------
        # Filter Short Coordination Episodes
        # ---------------------------------------------
        gaze['coord_final'] = gaze['coord_holes'].copy()
        minimum_duration = 26  # 25 FPS, equivalent to 1 second

        for i in np.arange(1, len(gaze['coord_holes']) - 1):
            if gaze['coord_holes'].iloc[i] == 1:
                if gaze['coord_holes'].iloc[i - 1] == 0:
                    idx_start = i
                if gaze['coord_holes'].iloc[i + 1] == 0:
                    idx_stop = i
                    if (idx_stop - idx_start) < minimum_duration:
                        gaze['coord_final'].iloc[idx_start:idx_stop + 1] = 0

        # ---------------------------------------------
        # Extract Coordination Episodes
        # ---------------------------------------------
        idx_start, idx_stop = get_st_sp(gaze['coord_final'])
        data_peaks = []

        for id_peak, (i_st, i_sp) in enumerate(zip(idx_start, idx_stop)):
            current_peak = gaze['coord_final'][i_st:i_sp]
            metrics = get_indexes(current_peak)

            time1 = FrameTimecode(timecode=int(i_st), fps=25)
            time2 = FrameTimecode(timecode=int(i_sp), fps=25)

            data_current_peak = [
                id_peak, i_st, i_sp,
                time1.get_timecode(precision=3, use_rounding=True),
                time2.get_timecode(precision=1, use_rounding=True),
                metrics
            ]

            data_peaks.append(data_current_peak)

        # ---------------------------------------------
        # Create and Save the Metrics DataFrame
        # ---------------------------------------------
        columns = ['coordination', 'start_frame', 'stop_frame', 'start_time', 'stop_time', 'duration']
        len_gaze = len(gaze)

        if not data_peaks:
            zeros = np.zeros(shape=(1, len(columns)))
            coord_pd = pd.DataFrame(zeros, columns=columns)
        else:
            coord_pd = pd.DataFrame(data_peaks, columns=columns)

        coord_pd['video_length'] = len_gaze
        save_path = get_new_filename(OUTDIR, subname)

        print(f"Saving: {subname} to {save_path}")
        coord_pd.to_csv(save_path)

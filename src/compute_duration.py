import argparse
import glob
import json
import random
from pathlib import Path
import os.path

import pandas as pd
import numpy as np
from tqdm import tqdm




def parse_args():
    parser = argparse.ArgumentParser(description="Compute EYE-C performance on hand-coding")

    parser.add_argument(
        "input_json",
        type=str,
        help="Folder where gazes_coordination jsons are stored",
    )
    parser.add_argument(
        "output_json_folder", type=str, help="Folder where to store the performance results"
    )
    parser.add_argument(
    "--min_duration",
    type=int,
    default=30,
    help="Minimum coordination duration (frames) treshold")

    # read and parse command line arguments
    args = parser.parse_args()


        # if only one is not None
    if (args.input_json is None) ^ (args.output_json_folder is None):
        parser.error("--input_json and --output_json_folder must be given together")


    return args


def main():

    args = parse_args()
    coords_file= Path(args.input_json)
    output_folder = Path(args.output_json_folder)
    min_duration_str = str(args.min_duration).replace(".", "")
    
    output_json_name = f"{coords_file.stem}_d{min_duration_str}.json"
    output_json_name = output_folder / output_json_name

    print(coords_file, min_duration_str)
    
    gaze = pd.read_json(coords_file)

    # GAZE COORDINATION with HOLES and MIN_DUR
    gaze['coordination']= gaze['coordination'].fillna(0)
    new_row = pd.DataFrame({'frame': -1, 'num_people': 0, 'eyes0': 0, 'gaze0': 0, 'bbox0': 0,'y_gaze0': 0, 'dist0': 0,
    'coordination0': 0, 'depth0':0,'eyes1': 0, 'gaze1': 0, 'bbox1': 0, 'y_gaze1': 0, 'dist1': 0,
    'coordination1': 0, 'depth1':0,'eyes2': 0, 'gaze2': 0, 'bbox2': 0, 'y_gaze2': 0, 'dist2': 0,
    'coordination2': 0, 'depth2':0, 'coordination': 0,'depth':0}, index=[0])

    gaze = pd.concat([new_row, gaze, new_row]).reset_index(drop = True)
    gaze['coord_holes'] = gaze['coordination']

    
    idx_start = 0
    idx_stop = 0

    for i in np.arange(1,len(gaze['coordination'])-1):
        if gaze.iloc[i, gaze.columns.get_loc("coordination")]==0:
            if gaze.iloc[i-1, gaze.columns.get_loc("coordination")]==1:
                idx_start = i
            if gaze.iloc[i+1, gaze.columns.get_loc("coordination")] ==1:
                idx_stop = i
                if len(gaze.iloc[(idx_start):(idx_stop), gaze.columns.get_loc("coordination")])<5:
                    gaze.iloc[(idx_start):(idx_stop+1), gaze.columns.get_loc("coord_holes")]= 1


    gaze['coord_final'] = gaze['coord_holes']


    idx_start = 0
    idx_stop = 0

    for i in np.arange(1,len(gaze['coord_holes'])-1):
        if gaze.iloc[i, gaze.columns.get_loc("coord_holes")]==1:
            if gaze.iloc[i-1, gaze.columns.get_loc("coord_holes")]==0:
                idx_start = i
            if gaze.iloc[i+1, gaze.columns.get_loc("coord_holes")] ==0:
                idx_stop = i
                if len(gaze.iloc[(idx_start):(idx_stop), gaze.columns.get_loc("coord_holes")])< args.min_duration:
                    gaze.iloc[(idx_start):(idx_stop+1), gaze.columns.get_loc("coord_final")]= 0
        
        
    out = pd.DataFrame(gaze)

    out.to_json(output_json_name)

if __name__ == "__main__":
    main()

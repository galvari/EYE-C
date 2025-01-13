import argparse
import glob
import json
import random
from pathlib import Path
import os.path

import pandas as pd
import numpy as np
from tqdm import tqdm
from scenedetect.frame_timecode import FrameTimecode





def get_indexes(x):
    duration = (len(x)/25) #questo e' in frames
    return(duration)


def get_st_sp(x):
    idx_peaks = x > 0 
    
    idx_peaks_diff = np.diff(idx_peaks)
    
    idx_start = np.where(idx_peaks_diff == 1)[0] +1 
    idx_stop = np.where(idx_peaks_diff == -1)[0] +1
    return(idx_start, idx_stop)




def parse_args():
    parser = argparse.ArgumentParser(description="Extract metrics from gazes_coordination.json files")

    parser.add_argument(
        "input_folder",
        type=str,
        help="Folder where json files are stored",
    )
    parser.add_argument(
        "output_metrics_folder", type=str, help="Folder where to store the output metrics"
    )
    parser.add_argument(
    "--min_duration",
    type=int,
    default=30,
    help="Minimum coordination duration (frames) treshold")


    # read and parse command line arguments
    args = parser.parse_args()


        # if only one is not None
    if (args.input_folder is None) ^ (args.output_metrics_folder is None):
        parser.error("--input_folder and --output_metrics_folder must be given together")

    
    return args



def main():
    args = parse_args()
    json_folder = Path(args.input_folder)
    jsons = os.listdir(json_folder)
    OUTDIR = Path(args.output_metrics_folder)
    json_subnames = np.unique([x.split('_')[0] for x in jsons])

    print("Loaded JSONs")
     
    for sub in jsons:
        subname = sub.split('_')[0]
        print(subname)
        json_filename_with_path = f'{json_folder}/{subname}_gazes_coordinations.json'
        gaze = pd.read_json(json_filename_with_path)



        # preprocess
        gaze['coordination']= gaze['coordination'].fillna(0)
        
        new_row = pd.DataFrame({'frame': -1, 'num_people': 0, 'eyes0': 0, 'gaze0': 0, 'bbox0': 0, 'y_gaze0': 0, 'dist0': 0,
        'coordination0': 0, 'eyes1': 0, 'gaze1': 0, 'bbox1': 0, 'y_gaze1': 0, 'dist1': 0,
        'coordination1': 0, 'eyes2': 0, 'gaze2': 0, 'bbox2': 0, 'y_gaze2': 0, 'dist2': 0,
        'coordination2': 0, 'coordination': 0}, index=[0])

        gaze = pd.concat([new_row, gaze, new_row]).reset_index(drop = True)
        gaze['coord_holes'] = gaze['coordination']
        
        # fill holes
        idx_start = 0
        idx_stop = 0

        for i in np.arange(1,len(gaze['coordination'])-1):
            if gaze.iloc[i, 20]==0:
                if gaze.iloc[i-1, 20]==1:
                    idx_start = i
                if gaze.iloc[i+1, 20] ==1:
                    idx_stop = i
                    if len(gaze.iloc[(idx_start):(idx_stop), 20])<5:
                        gaze.iloc[(idx_start):(idx_stop+1), 21]= 1


        gaze['coord_final'] = gaze['coord_holes']

        
        # select episodes
        idx_start = 0
        idx_stop = 0

        for i in np.arange(1,len(gaze['coord_holes'])-1):
            if gaze.iloc[i, 21]==1:
                if gaze.iloc[i-1, 21]==0:
                    idx_start = i
                if gaze.iloc[i+1, 21] ==0:
                    idx_stop = i
                    if len(gaze.iloc[(idx_start):(idx_stop), 21])< args.min_duration:
                        gaze.iloc[(idx_start):(idx_stop+1), 22]= 0



        idx_start, idx_stop = get_st_sp(gaze['coord_final'])

        data_peaks = []
        data_current_peak = []

        for id_peak, (i_st, i_sp) in enumerate(zip(idx_start, idx_stop)): #for each peak
            current_peak = gaze['coord_final'][i_st: i_sp] #extract peaks
            metrics = get_indexes(current_peak) #compute peak metrics

            time1 = FrameTimecode(timecode = int(i_st), fps = 25)
            time2 = FrameTimecode(timecode = int(i_sp), fps = 25)

            data_current_peak = [id_peak,(i_st), (i_sp),time1.get_timecode(precision = 1, use_rounding=True), time2.get_timecode(precision = 1, use_rounding=True), metrics]
        
            data_peaks.append(data_current_peak)

        coord_pd = pd.DataFrame(data_peaks)
        coord_pd.columns = ['coordination', 'start_frame', 'stop_frame', 'start_time', 'stop_time',
                                'duration']

        coord_pd.to_csv(f'{OUTDIR}/{subname}_metrics.csv')

if __name__ == "__main__":
    main()
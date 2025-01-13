import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from scenedetect.frame_timecode import FrameTimecode
import re
pd.options.mode.chained_assignment = None  # default='warn'

import os.path


params_list = ['dist08_z06','dist08_z09','dist08_z03']

for params in param_list:

    json_folder = f".../JSON_coordination/{params}"

    jsons = os.listdir(json_folder)

    OUTDIR = f'.../Coord_Metrics/{params}'
    # Ensure the directory exists
    os.makedirs(OUTDIR, exist_ok=True)

    json_subnames = np.unique([x.split('_')[1] for x in jsons])

    print('Found: ', json_subnames)

    def get_new_filename(base_path, subname):
        
        # check for multiple files for same subject
        counter = 1
        original_filename = f"{base_path}/{subname}_metrics.csv"
        new_filename = original_filename

        # Check if the file exists, and create a new filename with a counter suffix
        while os.path.exists(new_filename):
            new_filename = f"{base_path}/{subname}_{counter}_metrics.csv"
            counter += 1

        return new_filename


    def get_indexes(x):
        duration = (len(x)/25) 
        return(duration)

    def get_st_sp(x):
        idx_peaks = x > 0 
        idx_peaks = idx_peaks.astype(int)
        
        idx_peaks_diff = np.diff(idx_peaks)
        
        idx_start = np.where(idx_peaks_diff == 1)[0] +1 
        idx_stop = np.where(idx_peaks_diff == -1)[0] +1 
        return(idx_start, idx_stop)

    for sub in jsons:
        print('Extracting: ', sub)
            # Use regular expression to find the pattern 'lab' followed by numbers
        match = re.search(r'lab\d+', sub)
        if match:
            subname = match.group()  # This will be 'lab684' for your example
        else:
            subname = sub.split('_')[0]

        json_filename_with_path = f'{json_folder}/{sub}'
        gaze = pd.read_json(json_filename_with_path)


        # preprocess
        gaze['coordination']= gaze['coordination'].fillna(0)

        new_row = pd.DataFrame({'frame': -1, 'num_people': 0, 'eyes0': 0, 'gaze0': 0, 'bbox0': 0, 'y_gaze0': 0, 'dist0': 0,
            'coordination0': 0, 'eyes1': 0, 'gaze1': 0, 'bbox1': 0, 'y_gaze1': 0, 'dist1': 0,
            'coordination1': 0, 'eyes2': 0, 'gaze2': 0, 'bbox2': 0, 'y_gaze2': 0, 'dist2': 0,
            'coordination2': 0, 'coordination': 0}, index=[0])
        gaze = pd.concat([new_row, gaze, new_row]).reset_index(drop = True)


        # Fill small holes
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
                
      
                        
                
        ## filter short episodes               
        gaze['coord_final'] = gaze['coord_holes'].copy()
        # Select episodes
        idx_start = 0
        idx_stop = 0
        
        minimum_duration = 26 # 25 frames per second

        for i in np.arange(1, len(gaze['coord_holes']) - 1):
            if gaze['coord_holes'].iloc[i] == 1:
                if gaze['coord_holes'].iloc[i - 1] == 0:
                    idx_start = i
                if gaze['coord_holes'].iloc[i + 1] == 0:
                    idx_stop = i
                    if (idx_stop - idx_start) < minimum_duration:  # Adjusted for clarity; it means the episode is shorter than 30 frames
                        gaze['coord_final'].iloc[idx_start:idx_stop + 1] = 0



        idx_start, idx_stop = get_st_sp(gaze['coord_final'])

        data_peaks = []
        data_current_peak = []

        for id_peak, (i_st, i_sp) in enumerate(zip(idx_start, idx_stop)): #for each peak
            current_peak = gaze['coord_final'][i_st: i_sp] #extract peaks
            metrics = get_indexes(current_peak) #compute peak metrics

            time1 = FrameTimecode(timecode = int(i_st), fps = 25)
            time2 = FrameTimecode(timecode = int(i_sp), fps = 25)

            data_current_peak = [id_peak,(i_st), (i_sp),time1.get_timecode(precision = 3, use_rounding=True), time2.get_timecode(precision = 1, use_rounding=True), metrics]
        
            data_peaks.append(data_current_peak)


        # Check if data_peaks is empty
        len_gaze = len(gaze)
        columns = ['coordination', 'start_frame', 'stop_frame', 'start_time', 'stop_time', 'duration']

        # Check if data_peaks is an empty list
        if data_peaks == []:
            # If data_peaks is empty, create an empty DataFrame with desired columns
            zeros = np.zeros(shape=(1,len(columns)))
            coord_pd = pd.DataFrame(zeros,columns=columns)
            coord_pd['video_length'] = len_gaze
        else:
            # If data_peaks is not empty, create DataFrame and add 'video_length' column
            coord_pd = pd.DataFrame(data_peaks, columns=columns)
            coord_pd['video_length'] = len_gaze

        # If data_peaks is empty, fill the DataFrame with zeros
        if not data_peaks:
            coord_pd[columns] = 0
            coord_pd['video_length'] = len(gaze)

        coord_pd['video_length']= len(gaze)
        save_path = get_new_filename(OUTDIR, subname)
        print(f'Saving: {subname, save_path}')
        coord_pd.to_csv(save_path)
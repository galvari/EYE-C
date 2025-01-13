import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import re
import os.path

params_list = ['dist08_z06', 'dist08_z03', 'dist08_z09']

def dist_max(row):
    if row['dist1_mean'] > row['dist0_mean']:
        return row['dist1_mean'], row['dist1_std']
    else:
        return row['dist0_mean'], row['dist0_std']

def dist_min(row):
    if row['dist1_mean'] < row['dist0_mean']:
        return row['dist1_mean'], row['dist1_std']
    else:
        return row['dist0_mean'], row['dist0_std']

def get_number_of_elements(list):
    count = 0
    for element in list:
        count += 1
    return int(count)

# ADD DISTS
for params in params_list:
    json_folder = f".../JSON_coordination/{params}"
    jsons = os.listdir(json_folder)
    json_subnames = np.unique([x.split('_')[1] for x in jsons])
    print(json_subnames)
    print('Sample: ', len(json_subnames))

    metric_folder = f'.../Coord_Metrics/{params}'
    OUTDIR = '.../ADOS_Pietro/Dataset'

    dists = []

    for sub in jsons:
        match = re.search(r'lab\d+', sub)
        if match:
            subname = match.group()
        else:
            subname = sub.split('_')[0]

        json_path = f'{json_folder}/{sub}'
        gaze = pd.read_json(json_path)

        dist0_mean = gaze.dist0.mean(skipna=True)
        dist0_std = gaze.dist0.std(skipna=True)
        dist1_mean = gaze.dist1.mean(skipna=True)
        dist1_std = gaze.dist1.std(skipna=True)
        dur = int(gaze.frame.iloc[-1])

        dists_current = [subname, dist0_mean, dist0_std, dist1_mean, dist1_std, dur]
        dists.append(dists_current)

    # DIST DATAFRAME
    dists_pd = pd.DataFrame(dists)
    dists_pd.columns = ['subject', 'dist0_mean', 'dist0_std', 'dist1_mean', 'dist1_std', 'frames']
    dists_pd.subject = dists_pd.subject.astype(str)
    dists_pd = dists_pd.groupby('subject').agg({
        'dist0_mean': 'mean',
        'dist0_std': 'mean',
        'dist1_mean': 'mean',
        'dist1_std': 'mean',
        'frames': 'sum'
    }).reset_index()

    #################### DEAL WITH MULTIPLE FILES #########################
    # Step 1: List all files and group by subject name
    file_groups = {}
    for filename in os.listdir(metric_folder):
        if filename.endswith("_metrics.csv") and 'merged' not in filename:
            subjectname = filename.split('_')[0]
            if subjectname not in file_groups:
                file_groups[subjectname] = []
            file_groups[subjectname].append(filename)

    # Step 2: Process each group of files
    for subjectname, filenames in file_groups.items():
        print(f'Merging metrics of {subjectname}...')
        merged_df = pd.DataFrame()
        total_video_length = 0

        # Merge files and sum 'video_length'
        for file in filenames:
            print('Elaborating: ', file)
            df = pd.read_csv(os.path.join(metric_folder, file))
            if 'video_length' in df.columns:
                if not df['video_length'].isnull().all():
                    total_video_length += df['video_length'].iloc[0]
                else:
                    print(f"Warning: 'video_length' column not found in {file}")
                    total_video_length += 0
            # Concatenate dataframes
            merged_df = pd.concat([merged_df, df], ignore_index=True)

        # Update 'video_length' to the total sum
        merged_df['video_length'] = total_video_length

        # Save the merged DataFrame
        merged_df.to_csv(os.path.join(metric_folder, f"{subjectname}_merged_metrics.csv"), index=False)

    ########### GENERATE GAZE DATASET ###################

    subjects = os.listdir(metric_folder)

    # GENERATE COORD_DATASET

    coord_metrics = []
    fps = 25

    for sub in subjects:
        print(f'Elaborating {sub} metrics...')
        if 'merged' in sub:
            subname = sub.split('_')[0]
            csv_filename_with_path = f'{metric_folder}/{subname}_merged_metrics.csv'
            metric = pd.read_csv(csv_filename_with_path)

            total_len = int(dists_pd.frames[dists_pd.subject == subname].iloc[0]) / fps
            video_len = metric['video_length'].iloc[0] / fps
            frame = (total_len / 4)
            time_current = []

            class coord_mean:
                duration = []
                number = []

            dur = np.mean(metric.duration)
            num = metric['coordination'].count()

            coord_current = [subname, dur, num, video_len, frame]

            # Initialize list to store start and stop frames for calculating time between episodes
            start_stop_frames = metric[['start_frame', 'stop_frame']].sort_values(by='start_frame').values

            # Calculate the time between episodes
            time_between_episodes = np.diff(start_stop_frames.flatten())[1::2] / fps

            # Calculate average and standard deviation of time between episodes
            if len(time_between_episodes) > 0:
                avg_time_between = np.mean(time_between_episodes)
                std_time_between = np.std(time_between_episodes)
            else:
                avg_time_between = 0
                std_time_between = 0

            # Initialize counts for each quarter
            n_q1, n_q2, n_q3, n_q4 = 0, 0, 0, 0

            while frame <= total_len:
                for idx, row in metric.iterrows():
                    if row['stop_frame'] <= frame * fps and row['stop_frame'] >= (frame - (total_len / 4)) * fps:
                        time_current.append(row['duration'])
                        if len(time_current) == 0:
                            time_current.append(0)

                        # Count the episodes in each quarter
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

            # Append coord_mean.duration and coord_mean.number to coord
            coord = coord_current + coord_mean.duration + coord_mean.number
            coord.extend([avg_time_between, std_time_between])  # Remove duplicate counts here
            coord_metrics.append(coord)


        else:
            print('No merged file:', sub)

    # COORD DATAFRAME
    coord_pd = pd.DataFrame(coord_metrics)
    coord_pd.columns = ['subject', 'av_len', 'num', 'video_len', 'quarter_len', 'len_q1', 'len_q2', 'len_q3', 'len_q4', 'n_q1', 'n_q2', 'n_q3', 'n_q4', 'avg_time_between', 'std_time_between']

    # Convert quarters from frames to seconds
    coord_pd['quarter_len'] = coord_pd['quarter_len']

    # Compute frequencies using the converted values in seconds
    coord_pd['freq'] = coord_pd['num'] / coord_pd['video_len']
    coord_pd['freq_q1'] = coord_pd['n_q1'] / coord_pd['quarter_len']
    coord_pd['freq_q2'] = coord_pd['n_q2'] / coord_pd['quarter_len']
    coord_pd['freq_q3'] = coord_pd['n_q3'] / coord_pd['quarter_len']
    coord_pd['freq_q4'] = coord_pd['n_q4'] / coord_pd['quarter_len']

    coord_pd.subject = coord_pd.subject.astype(str)

    # CSV GAZE-COORD
    GazeCoordDataset = pd.merge(coord_pd, dists_pd, on='subject')
    GazeCoordDataset['dist_min_mean'], GazeCoordDataset['dist_min_std'] = zip(*GazeCoordDataset.apply(lambda row: dist_min(row), axis=1))
    GazeCoordDataset['dist_max_mean'], GazeCoordDataset['dist_max_std'] = zip(*GazeCoordDataset.apply(lambda row: dist_max(row), axis=1))
    GazeCoordDataset['subject'] = GazeCoordDataset['subject'].str.extract('(\d+)')
    GazeCoordDataset['subject'] = pd.to_numeric(GazeCoordDataset['subject'], errors='coerce')
    GazeCoordDataset.to_csv(f'{OUTDIR}/GazeCoordDataset_{params}.csv', index=False)



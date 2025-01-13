import argparse
import glob
import json
import random
from pathlib import Path
import os.path

import pandas as pd
import numpy as np
from tqdm import tqdm



def dist_max (row):
   if row['dist1'] > row['dist0']:
       return row['dist1']

   else:
       return row['dist0']


def dist_min (row):
   if row['dist1'] < row['dist0']:
       return row['dist1']

   else:
       return row['dist0']


def get_number_of_elements(list):
    count = 0
    for element in list:
        count += 1
    return int(count)




def parse_args():
    parser = argparse.ArgumentParser(description="Extract final dataset with all metrics and dist")

    parser.add_argument(
        "input_json_folder",
        type=str,
        help="Folder where dist are stored",
    )
    parser.add_argument(
    "input_metrics_folder",
    type=str,
    help="Folder where metrics are stored",
    )
    parser.add_argument(
    "input_dataset",
    type=str,
    help="Folder where datset (excel) is stored",
    )
    parser.add_argument(
        "output_dataset_folder", type=str, help="Folder where to store the final dataset"
    )

    # read and parse command line arguments
    args = parser.parse_args()


        # if only one is not None
    if (args.input_folder is None) ^ (args.input_dataset is None):
        parser.error("--input_folder and --input_dataset must be given together")


    return args


def main():
    args = parse_args()
    json_folder = Path(args.input_json_folder)
    jsons = os.listdir(json_folder)

    print("Loaded jsons")




    # ADD DISTS

    dists = []

    for sub in jsons:

        subname = sub.split('_')[0]

        json_path = f'{json_folder}/{subname}_gazes_coordinations.json'

        gaze = pd.read_json(json_path)

        dist0 = np.nanmean(gaze.dist0)
        dist1 = np.nanmean(gaze.dist1)
        dur = int(gaze.frame.iloc[-1])

        dists_current = [subname, dist0, dist1, dur]
        dists.append(dists_current)

    # DIST DATAFRAME

    dists_pd = pd.DataFrame(dists)
    dists_pd.columns = ['subject', 'dist0', 'dist1', 'frames']
    dists_pd.subject = dists_pd.subject.astype(str)







    gaze_pd = pd.read_excel(Path(args.input_dataset))
    gaze_pd.subject = gaze_pd.subject.astype(str)


    metric_folder = Path(args.input_metrics_folder)
    subjects = os.listdir(metric_folder)

    OUTDIR = Path(args.output_dataset_folder)



    # GENERATE COORD_DATASET

    coord_metrics =[]

    for sub in subjects:
        
        subname = sub.split('_')[0]
        csv_filename_with_path = f'{metric_folder}/{subname}_metrics.csv'
        metric = pd.read_csv(csv_filename_with_path)

        len = int(dists_pd.frames[dists_pd.subject == subname])


        frame = (len/4)
        time_current = []

        class coord_mean():
            duration = []
            number = []

        dur = np.mean(metric.duration)
        num = metric['coordination'].count()

        coord_current = [subname, dur, num]

        while frame <= len:

            for idx, row in metric.iterrows():

                if metric.stop_frame[idx] <= frame and metric.stop_frame[idx] >= frame -(len/4):
                    time_current.append(metric.duration[idx])
                    if get_number_of_elements(time_current) == 0:
                        time_current.append(0)

            coord_mean.duration.append(np.mean(time_current))
            coord_mean.number.append(get_number_of_elements(time_current))
            time_current = []
            frame += len/4

        coord = coord_current + coord_mean.duration + coord_mean.number
        coord_metrics.append(coord)

    # COORD DATAFRAME

    coord_pd = pd.DataFrame( coord_metrics)
    coord_pd.columns= ['subject', 'av_len','num','len1','len2','len3','len4','n1','n2','n3','n4']
    coord_pd.subject = coord_pd.subject.astype(str)

    coord_pd.to_csv(f'{OUTDIR}/CoordDataset.csv')

    # CSV GAZE-COORD-DISTS
    coord_dist = pd.merge(coord_pd, dists_pd, on='subject')
    GazeCoordDataset = pd.merge(gaze_pd, coord_dist, on='subject')

    GazeCoordDataset['freq'] = GazeCoordDataset.num/GazeCoordDataset.length
    GazeCoordDataset['dist_min'] = GazeCoordDataset.apply(lambda row: dist_min(row), axis=1)
    GazeCoordDataset['dist_max'] = GazeCoordDataset.apply(lambda row: dist_max(row), axis=1)
    GazeCoordDataset.to_csv(f'{OUTDIR}/GazeCoordDataset.csv', index = False)

if __name__ == "__main__":
    main()


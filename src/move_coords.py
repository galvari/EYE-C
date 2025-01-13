import os
from os import path
import shutil
import argparse
import glob
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Move coordinations JSON to selected folder.")
    parser.add_argument("input_folder", type=str, help="Input coordinations folder")
    parser.add_argument("output_folder", type=str, help="Where to store the coordinations")  

    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    source_dir = os.listdir(args.input_folder)
    target_dir = args.output_folder
    
    #files = [i for i in os.listdir(target_dir) if i.endswith("coordinations.json") and path.isfile(path.join(target_dir, i))]
    #for f in files:
         #shutil.copy(path.join(target_dir, f), target_fir)

    for files in source_dir:    
        if files.endswith('_coordinations.json'):
            shutil.move(source_dir, target_dir)

    
    


    #os.chdir(source_dir)

    #for file in os.listdir("."):
        #if os.path.isfile(file) and file.endswith("coordinations.json"):
            #shutil.move(os.path.join(source_dir, file), target_dir)

            

if __name__ == "__main__":
    main()

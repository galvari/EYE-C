# EYE-C: Eye-Contact Robust Detection in Unconstrained Interactions

## About
EYE-C is a robust multi-person eye-contact detection framework designed for analyzing interactions in wild clinical settings using a single video camera. The application integrates key components from [OpenPose](https://ieeexplore.ieee.org/document/8765346) and [Gaze360](https://gaze360.csail.mit.edu/) to achieve reliable head detection and gaze estimation.

This repository includes:
- **OpenPose**: for head detection.
- **Gaze360**: for gaze vector estimation.
- **EYE-C**: for eye-contact prediction based on gaze vectors.

### Reference
For details about the implementation and validation, refer to our paper:
[EYE-C: Eye-Contact Robust Detection and Analysis during Unconstrained Child-Therapist Interactions in the Clinical Setting of Autism Spectrum Disorders](https://www.mdpi.com/2076-3425/11/12/1555)

If you use this code, please cite our paper:
```bibtex
@article{alvari2021eye,
  title={EYE-C: eye-contact robust detection and analysis during unconstrained child-therapist interactions in the clinical setting of autism spectrum disorders},
  author={Alvari, Gianpaolo and Coviello, Luca and Furlanello, Cesare},
  journal={Brain Sciences},
  volume={11},
  number={12},
  pages={1555},
  year={2021},
  publisher={MDPI}
}
```

---

## Repository Structure
- **src/**: Contains the main scripts for running EYE-C along with the OpenPose and Gaze360 modules.
  - `extract_gaze.py`: Generates gaze estimation JSON files and renders the video with gaze vectors.
  - `visualize_faces.py`: Visualizes head bounding boxes detected by OpenPose.
  - `estimate_coordination.py`: Calculates gaze coordination between subjects.
  - `coord_extraction.py`: Extracts metrics from gaze coordination JSON files to identify eye-contact episodes and compute durations.
  - `extract_bbox_size.py`: Extracts bounding box sizes from OpenPose keypoint files and saves them in a CSV file.
  - `generate_metrics.py`: Extracts metrics for gaze coordination episodes.
  - `generate_metrics_dataset.py`: Combines multiple metrics files into a single dataset.
  - `generate_coordination_dataset.py`: Creates a dataset of gaze coordination episodes.
  - `metrics_dataset.py`: Generates a comprehensive dataset with coordination metrics.
- **Run Shell Files**: A set of shell scripts to automate the execution of the EYE-C pipeline from the terminal.

---

## Usage Guide
To run the EYE-C pipeline, follow these steps:

### Step 1: Run OpenPose
Run OpenPose on your videos to generate keypoint JSON files for each frame. Ensure the keypoint files are saved in a designated folder.


### Step 2: Run `extract_gaze.py`
This script processes the keypoint files to estimate gaze directions for each subject and renders a video with gaze vectors overlaid.
```
Arguments:
    input_folder: Folder containing keypoint files.
    input_video: Path to the input video.
    output_video_folder: Folder to save the rendered video.
    output_json_folder: Folder to save the gaze JSON files.
    --use_cuda: Use GPU for processing.
    --model_weights: Path to the Gaze360 model weights.
    --maximize_boxes: Use the largest bounding box in the sequence for each subject.
    --enlarge_boxes: Enlarge the bounding box to improve head cropping.
```

### Step 2 (Optional): Run `visualize_faces.py`
This script can be used to verify the head detection by rendering a video with bounding boxes around detected heads.
```
Arguments:
    input_folder: Folder containing OpenPose keypoint files.
    input_video: Path to the input video.
    output_folder: Folder to save the output video.
```

### Step 3: Run `estimate_coordination.py`
This script calculates gaze coordination between subjects by identifying whether they are looking at each other in each frame.
```
Arguments:
    input_coords: Input gaze file (used for naming the output file).
    --input_video: Path to the input video (optional).
    --output_folder: Folder to save the output video (optional).
    --coordination_factor: Scaling factor to check for coordination (default: 1).
```
---
## Extras:

### Run `coord_extraction.py`
Extracts metrics from gaze coordination JSON files to identify coordination episodes, compute durations, and save the results as CSV files with timestamps and frames.
```
Arguments:
    input_folder: Folder containing gaze coordination JSON files.
    output_metrics_folder: Folder to save the extracted metrics CSV files.
    --min_duration: Minimum duration threshold for coordination episodes (default: 30 frames).
```


### Run `extract_bbox_size.py`
Extracts bounding box sizes from OpenPose keypoint files and saves them in a CSV file for further analysis.
```
Arguments:
    input_folder: Folder containing OpenPose keypoint files.
    input_video: Path to the input video.
    output_folder: Folder to save the output CSV file.
```

### Run `generate_metrics.py`
This script extracts coordination metrics for each video, such as the number and duration of coordination episodes.
```
Arguments:
    input_folder: Folder containing gaze coordination JSON files.
    output_folder: Folder to save the metrics CSV files.
```

### Run `generate_metrics_dataset.py`
This script combines coordination metrics from multiple videos into a single dataset for analysis.
```
Arguments:
    input_json_folder: Folder containing distance JSON files.
    input_metrics_folder: Folder containing coordination metrics CSV files.
    input_dataset: Path to the input dataset (Excel file).
    output_dataset_folder: Folder to save the final dataset.
```

### Run `generate_coordination_dataset.py`
Generates a dataset containing detailed gaze coordination metrics for each subject.
```
Arguments:
    input_folder: Folder containing keypoint files.
    output_metrics_folder: Folder to save the output metrics.
    --min_duration: Minimum duration threshold for coordination episodes (default: 30 frames).
```


---

## References
```bibtex
@article{alvari2021eye,
  title={EYE-C: eye-contact robust detection and analysis during unconstrained child-therapist interactions in the clinical setting of autism spectrum disorders},
  author={Alvari, Gianpaolo and Coviello, Luca and Furlanello, Cesare},
  journal={Brain Sciences},
  volume={11},
  number={12},
  pages={1555},
  year={2021},
  publisher={MDPI}
}

@inproceedings{kellnhofer2019gaze360,
  title={Gaze360: Physically unconstrained gaze estimation in the wild},
  author={Kellnhofer, Petr and Recasens, Adria and Stent, Simon and Matusik, Wojciech and Torralba, Antonio},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6912--6921},
  year={2019}
}

@article{8765346,
  author = {Z. {Cao} and G. {Hidalgo Martinez} and T. {Simon} and S. {Wei} and Y. A. {Sheikh}},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title = {OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
  year = {2019}
}
```

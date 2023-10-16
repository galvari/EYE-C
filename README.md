# EYE-C
A Gaze360 implementation to extract Eye-Contact events from wild videos

This implementation makes use of the original Gaze360 code, that can be found at https://github.com/erkil1452/gaze360

## Usage
run openpose on the videos to extract the keypoint files

optional: run `visualize_faces.py` to visualize the bbox generated from the keypoint files.
```
Arguments:
    input_folder: folder where keypoints file for the video are stored
    input_video
    output_folder
```

run `extract_gaze.py` to generate gaze json files containing the gaze for each subject in each frame. Also renders the video with gazing arrows.
```
Arguments:
    input_folder: folder where keypoints file for the video are stored
    input_video
    output_video_folder: where to store the rendered video
    output_json_folder: folder where to store the output json containg gazes
    --use_cuda: whether to use GPU
    --model_weights: file containing the model weights
    --maximize_boxes: whether to use the maximum box size of the sequence for each subject
    --enlarge_boxes: whether to enlarge the box used to crop the heads
```

run `estimate_coordination.py` to generate coordination json files for each frame of the video. Optionally render the video with printed distances.
```
Arguments:
    input_coords: input gaze file (used also for naming the output file)
    --input_video: input video (optional)
    --output_folder: where to store the output video (optional)
    --coordination_factor: scaling factor to check for coordination, default is 1
```

_______

Reference paper:

Petr Kellnhofer*, Adrià Recasens*, Simon Stent, Wojciech Matusik, and Antonio Torralba. “Gaze360: Physically Unconstrained Gaze Estimation in the Wild”. IEEE International Conference on Computer Vision (ICCV), 2019.

Gianpaolo Alvari*, Luca Coviello, and Cesare Furlanello. "EYE-C: Eye-Contact Robust Detection and Analysis during Unconstrained Child-Therapist Interactions in the Clinical Setting of Autism Spectrum Disorders". Brain Sci. 2021.

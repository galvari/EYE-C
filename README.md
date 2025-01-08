# EYE-C: Eye-Contact Robust Detection in Unconstrained Interactions

## About
A Openpose+Gaze360 implementation to extract multi-person Eye-Contact events from wild clinical videos, via a single videocamera.
This application makes use of the original Openpose and Gaze360 code, that can be found at https://github.com/CMU-Perceptual-Computing-Lab/openpose and https://github.com/erkil1452/gaze360 respectively.

EYE-C is implemented in 3 connected modules:
* Openpose: headboxes detection
* Gaze360: gaze vectors estimation
* EYE-C: eye-contact prediction
  
Details about the implementation and validation can be found in the original paper:
[EYE-C: Eye-Contact Robust Detection and Analysis during Unconstrained Child-Therapist Interactions in the Clinical Setting of Autism Spectrum Disorders](https://www.mdpi.com/2076-3425/11/12/1555)

Please if you use our code cite our paper as:
Alvari, G., Coviello, L., & Furlanello, C. (2021). EYE-C: eye-contact robust detection and analysis during unconstrained child-therapist interactions in the clinical setting of autism spectrum disorders. Brain Sciences, 11(12), 1555.


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


## Usage
1. run openpose on the videos to extract the keypoint files

optional: run `visualize_faces.py` to visualize the bbox generated from the keypoint files.
```
Arguments:
    input_folder: folder where keypoints file for the video are stored
    input_video
    output_folder
```

2. run `extract_gaze.py` to generate gaze json files containing the gaze for each subject in each frame. Also renders the video with gazing arrows.
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

3. run `estimate_coordination.py` to generate coordination json files for each frame of the video. Optionally render the video with printed distances.
```
Arguments:
    input_coords: input gaze file (used also for naming the output file)
    --input_video: input video (optional)
    --output_folder: where to store the output video (optional)
    --coordination_factor: scaling factor to check for coordination, default is 1
```

_______

References:

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
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={6912--6921},
  year={2019}
}

@article{8765346,
  author = {Z. {Cao} and G. {Hidalgo Martinez} and T. {Simon} and S. {Wei} and Y. A. {Sheikh}},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title = {OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
  year = {2019}
}


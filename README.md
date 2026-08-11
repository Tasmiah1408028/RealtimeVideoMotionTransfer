# Real-time Video Motion Transfer

## Description of the repository:
This is a repo containing files related to keypoint prediction and video generation using VRNN in the First Order Motion Model (FOMM) pipeline. 

## Directory Contents:

The folder contains Jupyter notebooks that can be used to generate predictions using VRNN in the FOMM pipeline in either reconstruction or transfer mode for the Bair dataset.
The naming convention followed for the notebooks is:
"Full_Pipeline_VRNN\_Bair\_{Mode}_mode" where the Mode can be reconstruction or transfer.

The Training_Prediction subfolder contains the following:

1. The subfolder FOMM contains Bair data files and various functions related to FOMM inference.
The files are partially sourced from the original FOMM github:
https://github.com/AliaksandrSiarohin/first-order-model.

2. The subfolder PREDICTOR contains prediction functions using VRNN.

The config subfolder contains the yaml file for the Bair dataset.

The checkpoints subfolder contains the trained VRNN keypoins prediction models using prediction horizons of 15 for VRNN.
Checkpoints for VRNN are named as "VRNN\_5001videos_bair_{# input frames}_{# output frames}" where {# input frames} and {# output frames} is 15 here.

The log subfolder is the directory for saving generated videos.

The two pickle files are the keypoints corresponding to 256 Bair videos during inference for source image and driving video frames.

To calculate JEDi, JEDi.py can be used after generating the videos and some trained models need to be downloaded in "videojedi_models" folder as instructed.

3. To avail the source image for generating videos in reconstruction mode and to evaluate the quality of generated videos, the ground truth videos can be found under this google drive link.
https://drive.google.com/drive/folders/1Zn9ANeLadblbuScp_t_WmmViR85o-CVo?usp=sharing

To run this file in the attached Jupyter notebooks, please copy the videos to the following path "Training_Prediction/FOMM/datasets/bair/test/".

## Checkpoints for the FOMM model and keypoints 
Checkpoints for the FOMM model trained on the Bair dataset can be found under this google drive link. 
https://drive.google.com/drive/folders/1pachVtWHibzDi3E61jUmqFfz2hVxA1GX?usp=drive_link.

This file has been sourced using the link in the original FOMM github:
https://github.com/AliaksandrSiarohin/first-order-model.

To run this file in the attached Jupyter notebooks, please copy the checkpoint file to the following path "Training_Prediction/FOMM/Trained_Models/".

The keypoints corresponding to 5001 Bair videos which can be used to train the VRNN can be found with the same google drive link.

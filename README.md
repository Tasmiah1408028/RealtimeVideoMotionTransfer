# Motion-Transfer-Keypoints-Prediction

## Description of the repository:
This is a repo containing files related to keypoint prediction and video generation using VRNN in the First Order Motion Model (FOMM) pipeline. 

## Directory Contents:

The folder contains Jupyter notebooks that can be used to generate predictions using VRNN in the FOMM pipeline in either reconstruction or transfer mode for the VoxCeleb dataset.
The naming convention followed for the notebooks is:
"Full_Pipeline_VRNN\_VoxCeleb\_{Mode}_mode" where the Mode can be reconstruction or transfer.

The Training_Prediction subfolder contains the following:

1. The subfolder FOMM contains Voxceleb data files and various functions related to FOMM inference.
The files are partially sourced from the original FOMM github:
https://github.com/AliaksandrSiarohin/first-order-model.

2. The subfolder PREDICTOR contains prediction functions using VRNN.

The config subfolder contains the yaml file for the VoxCeleb dataset.

The checkpoints subfolder contains the trained VRNN keypoints prediction models using prediction horizons of 6 or 12 for VRNN.
Checkpoints for VRNN are named as "VRNN\_3883videos_vox_{# input frames}_{# output frames}" where {# input frames} and {# output frames} can be 6/12 indicates types of prediction.

The log subfolder is the directory for saving generated videos.

The two pickle files are the keypoints corresponding to 44 VoxCeleb videos during inference for source image and driving video frames.
## Checkpoints for the FOMM model and keypoints 
Checkpoints for the FOMM model trained on the VoxCeleb dataset can be found under this google drive link. 
https://drive.google.com/drive/folders/1pachVtWHibzDi3E61jUmqFfz2hVxA1GX?usp=drive_link.

This file has been sourced using the link in the original FOMM github:
https://github.com/AliaksandrSiarohin/first-order-model.

To run this file in the attached Jupyter notebooks, please copy the checkpoint file to the following path "Training_Prediction/FOMM/Trained_Models/".

The keypoints corresponding to 3883 VoxCeleb videos which can be used to train the RNN/VAE/VRNN can be found with the same google drive link.

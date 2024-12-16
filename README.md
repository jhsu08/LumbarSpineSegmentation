# LumbarSpineSegmentation
Elyas Amin, Jeffrey

## Overview
This repository provides scripts and tools for performing 3D segmentation of lumbar spine MRI images using state-of-the-art transformer-based models such as **SwinUNETR**. The goal is to process and segment high-resolution medical images efficiently while addressing the challenges of memory usage and computational costs.

## Requirements
**Python Version:** Requires **Python 3.11.x**.

### Dependencies
Install the required libraries:
```bash
pip install torch torchvision timm SimpleITK nibabel numpy scipy matplotlib scikit-learn monai 'monai[einops]' 'monai[itk]'
```
### Additional Tools
Git LFS (for handling large files such as pre-trained model weights):
```bash
brew install git-lfs
git lfs install
```

## Dataset
The dataset is provided by the SPIDER Challenge, consisting of sagittal T1 and T2 MRI images with varying resolutions:
- Standard Sagittal T1 and T2 Images:
Resolution ranges from 3.3 x 0.33 x 0.33 mm to 4.8 x 0.90 x 0.90 mm.
- Sagittal T2 SPACE Sequence Images:
Near-isotropic spatial resolution with voxel size 0.90 x 0.47 x 0.47 mm.
```bash
wget https://zenodo.org/api/records/10159290/files-archive
```
## Baseline Model
### Setup
Install the required tools for the baseline model:
```bash
brew install git-lfs
git lfs install
pip install git+https://github.com/DIAGNijmegen/Tiger.git@stable
```
### Description

The baseline model was trained using traditional CNN architectures. Preprocessing involved resizing images to isotropic spacing, downsampling for memory efficiency, and splitting the dataset into training and validation folds.

## Swin Model
### Pre-trained Weights
Download the pre-trained Swin model weights:
```bash
wget https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt
```
### Description
The SwinUNETR model is a transformer-based architecture designed for 3D medical image segmentation. This repository includes scripts for training and evaluating the SwinUNETR model on lumbar spine segmentation tasks.

### Key Features:
- Transformer-based Architecture: Incorporates Swin Transformer blocks for efficient and hierarchical feature extraction.
- Isotropic Resampling: MRI images are resampled to isotropic voxel spacing to standardize resolution.
- Sliding Window Inference: Used to handle large 3D volumes efficiently during inference.
- Customizable Pipeline: Easy integration of different datasets and models.

### Training and Evaluation
The training pipeline includes support for:
- Multi-GPU Training: Leverages NVIDIA GPUs for efficient computation.
- Early Stopping: Prevents overfitting by monitoring validation loss.
- Metrics: Dice Score and Dice Loss are used to evaluate segmentation performance.

To train the model:
1.	Edit the script with the correct paths to the training images and output directories.
2.	Run the script from the terminal using your preferred configuration.

Visualizing Results:
After training, results can be visualized by loading the trained model into the provided Jupyter notebook, which includes tools for overlaying segmentation masks on original images.
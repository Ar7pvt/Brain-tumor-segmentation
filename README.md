# 🧠 Brain MRI Segmentation

A deep learning application for segmenting brain tumors in MRI images using a U-Net neural network architecture.

## Overview

This project implements an automated brain MRI segmentation system that uses a trained U-Net model to identify and segment tumor regions in MRI scans. The application provides an easy-to-use interface built with Streamlit for segmenting new MRI images.

## Features

- **Pre-trained U-Net Model**: Uses a trained U-Net architecture for accurate brain tumor segmentation
- **Interactive Web Interface**: Built with Streamlit for easy image upload and prediction
- **Custom Metrics**: Implements Dice coefficient and Intersection over Union (IoU) metrics
- **Image Processing**: Automatic image resizing and normalization
- **Visual Output**: Side-by-side comparison of original and segmented images

## Project Structure

```
Brain-MRI-Segmentation/
├── App.py                          # Streamlit application
├── BRAIN1.ipynb                    # Jupyter notebook for model training/exploration
├── requirements.txt                # Python dependencies
├── Data/
│   ├── unet_brain_mri_seg.hdf5    # Pre-trained U-Net model
│   ├── kaggle_3m/                 # TCGA brain MRI dataset
│   └── lgg-mri-segmentation/      # LGG MRI segmentation dataset
├── Accuracy Graph.png              # Training accuracy visualization
├── Loss Graph.png                  # Training loss visualization
└── Unet_Architecture.png           # U-Net network architecture diagram
```

## Installation

### Prerequisites
- Python 3.11.x (recommended)
- Python 3.7-3.11 supported (TensorFlow 2.15 requirement)
- pip (Python package installer)
- Windows/macOS/Linux

### Setup Instructions

1. **Clone or download the repository**
   ```bash
   cd "Final year\Brain-MRI-Segmentation"
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   ```bash
   # On Windows PowerShell:
   .venv\Scripts\Activate.ps1
   
   # On Windows CMD:
   .venv\Scripts\activate.bat
   
   # On macOS/Linux:
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```


## Usage

### Quick Start

After installation and activation of the virtual environment:

```bash
# Ensure virtual environment is activated (you should see (.venv) in terminal)
# If not, activate it:
# Windows: .venv\Scripts\Activate.ps1
# Linux/Mac: source .venv/bin/activate

# Run the application
streamlit run App.py
```

The application will open in your web browser at `http://localhost:8501`

**Note**: Make sure your virtual environment is activated before running the app.

### Steps to Segment an MRI Image

1. Upload an MRI image (PNG or JPG format)
2. The uploaded image will be displayed
3. Click the **"🔍 Predict Segmentation"** button
4. View the original image and predicted segmentation side-by-side

## Model Details

### Architecture
- **Model Type**: U-Net Convolutional Neural Network
- **Input Size**: 256 × 256 pixels
- **Output**: Binary segmentation mask

### Custom Loss Functions and Metrics

The model uses custom metrics for training and evaluation:

- **Dice Coefficient**: Measures overlap between predicted and true segmentation
  ```
  Dice = (2 × Intersection + smooth) / (Union + smooth)
  ```

- **IoU (Intersection over Union)**: Also known as Jaccard Index
  ```
  IoU = (Intersection + smooth) / (Union - Intersection + smooth)
  ```

- **Dice Loss**: `-Dice Coefficient` used as the loss function

## Dependencies

Core packages and their versions:

- **numpy** (1.21.0 - 1.x): Numerical computing and array operations
- **streamlit** (≥1.28.0): Interactive web application framework
- **tensorflow** (2.15.0): Deep learning framework (Python 3.11 compatible)
- **keras** (2.15.0): High-level neural networks API
- **opencv-python** (≥4.5.0): Image processing and computer vision
- **matplotlib** (≥3.5.0): Data visualization and plotting
- **h5py** (≥3.7.0): HDF5 binary data format support
- **Pillow** (≥9.0.0): Python Imaging Library for image handling

## References

- **U-Net Paper**: Ronneberger, O., Fischer, P., & Brox, T. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). MICCAI 2015.
- **Dice Coefficient**: Sørensen–Dice coefficient for medical image segmentation evaluation
- **TCGA Dataset**: [The Cancer Genome Atlas](https://www.cancer.gov/tcga) - National Cancer Institute
- **Kaggle LGG Dataset**: [Brain MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)


## Acknowledgments

- U-Net architecture by Olaf Ronneberger et al.
- TCGA Research Network for the brain tumor dataset
- Kaggle community for dataset hosting and support




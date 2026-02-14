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

5. **Verify installation (optional)**
   ```bash
   # Test TensorFlow installation
   python -c "from tensorflow import keras; print('✓ TensorFlow OK')"
   
   # Check installed packages
   pip list | findstr "tensorflow streamlit"
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

### Running the Streamlit App

```bash
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

**Important**: TensorFlow 2.15.0 is required for Python 3.11 compatibility. Earlier TensorFlow versions (2.10-2.14) only support Python 3.7-3.10.

See [requirements.txt](requirements.txt) for complete specifications.

## Model Loading

The pre-trained model is loaded with custom objects registered via Keras serialization:

```python
model = load_model(model_path, custom_objects={
    'dice_coefficients_loss': dice_coefficients_loss,
    'iou': iou,
    'dice_coefficients': dice_coefficients,
    'jaccard_distance': jaccard_distance
}, compile=False)
```

## Performance

The model achieves strong performance on brain MRI segmentation tasks as evidenced by the included:
- Accuracy Graph (training accuracy over epochs)
- Loss Graph (training loss convergence)

## Troubleshooting

### Python Version Compatibility
- **Python 3.11+**: Use TensorFlow 2.15.0 (included in requirements.txt)
- **Python 3.10 or earlier**: TensorFlow 2.10+ will work
- If you encounter import errors, ensure you're using the virtual environment

### TensorFlow Import Errors
If you see `Import "tensorflow.keras" could not be resolved`:
1. **Reload VS Code window**: Press `Ctrl+Shift+P` → "Reload Window"
2. **Verify installation**: Run `pip show tensorflow keras` in terminal
3. **Test import**: Run `python -c "from tensorflow import keras; print('OK')"`
4. The editor warnings may persist due to IntelliSense cache, but the code will run correctly

### Virtual Environment Not Activated
If packages aren't found:
```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Model Loading Errors
- Ensure `unet_brain_mri_seg.hdf5` exists at: `Data/unet_brain_mri_seg.hdf5`
- Verify the model file is not corrupted (should be ~90MB)
- Check that TensorFlow 2.15.0 and Keras 2.15.0 are installed

### Image Upload Issues
- **Supported formats**: PNG, JPG/JPEG only
- **Recommended size**: Any size (will be automatically resized to 256×256)
- **Image quality**: Ensure MRI scan is clear and properly formatted
- **File size**: No strict limit, but keep under 10MB for best performance

### Performance Issues
- **GPU acceleration**: Recommended for faster predictions (CUDA-compatible NVIDIA GPU)
- **CPU mode**: TensorFlow will use CPU if no GPU available (slower but functional)
- **Memory**: Minimum 4GB RAM recommended
- **First prediction**: May take longer due to model initialization (~10-20 seconds)
- **Subsequent predictions**: Should be faster (~1-3 seconds)

### Warnings During Execution
You may see these warnings (safe to ignore):
```
I tensorflow/core/util/port.cc:113] oneDNN custom operations are on...
WARNING:tensorflow:From ... The name tf.losses.sparse_softmax_cross_entropy is deprecated...
```
These are informational and won't affect functionality.


## Future Enhancements

- Multi-class segmentation (tumor subtypes: necrosis, edema, enhancing tumor)
- Batch processing capabilities for multiple images
- 3D volume slicing and visualization
- Model retraining pipeline with new datasets
- Confidence score visualization and uncertainty estimation
- Export segmentation results (masks, overlays, reports)
- REST API for integration with other systems
- Docker containerization for easy deployment

## Datasets

The project uses two publicly available brain MRI datasets:

1. **TCGA Brain MRI Dataset (kaggle_3m)**: 
   - Contains brain MRI scans from The Cancer Genome Atlas
   - 110 patients with lower-grade glioma
   - FLAIR sequence images with genomic cluster data
   - Located in: `Data/kaggle_3m/`

2. **LGG MRI Segmentation Dataset**: 
   - Low-grade glioma brain tumor dataset
   - Pre-processed MRI slices with segmentation masks
   - Located in: `Data/lgg-mri-segmentation/`

## References

- **U-Net Paper**: Ronneberger, O., Fischer, P., & Brox, T. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). MICCAI 2015.
- **Dice Coefficient**: Sørensen–Dice coefficient for medical image segmentation evaluation
- **TCGA Dataset**: [The Cancer Genome Atlas](https://www.cancer.gov/tcga) - National Cancer Institute
- **Kaggle LGG Dataset**: [Brain MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

## License

This project is for educational purposes as part of a final year project. The datasets used are publicly available under their respective licenses. Please refer to the original dataset sources for usage restrictions and citations.

## Acknowledgments

- U-Net architecture by Olaf Ronneberger et al.
- TCGA Research Network for the brain tumor dataset
- Kaggle community for dataset hosting and support

## Contact & Support

For questions or issues, please refer to:
- Project documentation and code comments
- [BRAIN1.ipynb](BRAIN1.ipynb) notebook for detailed implementation and training examples
- TensorFlow/Keras documentation for deep learning concepts

---

**Note**: This is a final year academic project demonstrating brain tumor segmentation using deep learning techniques. Not intended for clinical use.



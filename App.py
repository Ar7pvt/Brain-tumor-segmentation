
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from keras.saving import register_keras_serializable
import h5py

# Set plot style
plt.style.use("ggplot")

#  Register custom functions for Keras
@register_keras_serializable()
def dice_coefficients(y_true, y_pred, smooth=100):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)

@register_keras_serializable()
def dice_coefficients_loss(y_true, y_pred, smooth=100):
    return -dice_coefficients(y_true, y_pred, smooth)

@register_keras_serializable()
def iou(y_true, y_pred, smooth=100):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    return (intersection + smooth) / (sum_ - intersection + smooth)

@register_keras_serializable()
def jaccard_distance(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    return -iou(y_true_flatten, y_pred_flatten)

# Streamlit App Title
st.title("🧠 Brain MRI Segmentation App")

# Load the U-Net model with error handling (HDF5 Only)
# model_path = "E:\\Final year\\unet_brain_mri_seg.hdf5"  # Adjust path as needed
model_path = "Data/unet_brain_mri_seg.hdf5"  # Adjust path as needed


try:
    model = load_model(model_path, custom_objects={
        'dice_coefficients_loss': dice_coefficients_loss,
        'iou': iou,
        'dice_coefficients': dice_coefficients,
        'jaccard_distance': jaccard_distance
    }, compile=False)  # Load without compiling
    
    # Recompile manually
    model.compile(optimizer="adam", loss=dice_coefficients_loss, metrics=[iou, dice_coefficients])


except Exception as e:
    st.error(f"❌ Failed to load model: {str(e)}")


# Constants for Image Processing
IM_HEIGHT = 256
IM_WIDTH = 256

# File Upload
file = st.file_uploader("📤 Upload MRI Image (PNG/JPG)", type=["png", "jpg"])
print("file)", file)

if file is not None:
    st.header("Uploaded Image:")
    st.image(file, caption="Uploaded MRI Image", use_container_width=True)

    # Convert uploaded file to OpenCV image
    content = file.getvalue()
    image = np.asarray(bytearray(content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Preprocess Image
    img_resized = cv2.resize(image, (IM_HEIGHT, IM_WIDTH))
    img_normalized = img_resized / 255.0  # Normalize pixel values
    img_input = np.expand_dims(img_normalized, axis=0)  # Add batch dimension
    
    if st.button("🔍 Predict Segmentation"):
        with st.spinner("Processing... 🔄"):
            pred_img = model.predict(img_input)[0]  # Shape: (256, 256, 1)
            pred_img = np.squeeze(pred_img)  # Shape: (256, 256)
            
            # Threshold prediction for binary mask
            threshold = 0.5
            binary_mask = (pred_img > threshold).astype(np.uint8) * 255

            # Display in two columns
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(img_resized, use_container_width=True)

            with col2:
                st.subheader("Predicted Segmentation")
                st.image(binary_mask, use_container_width=True)




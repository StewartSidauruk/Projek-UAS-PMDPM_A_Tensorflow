import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the pre-trained model
model_path = r'C:\Users\stewa\Downloads\model_mobilenet_tubes.h5'
model = load_model(model_path)

# Define class names
class_names = ["Blackberry", "Blueberry", "Strawberry"]

# Function to preprocess and classify image
def classify_image(image_path):
    try:
        # Load and preprocess the image
        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)

        # Predict using the model
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])  # Apply softmax for probability

        # Get class with highest confidence
        class_idx = np.argmax(result)
        confidence_scores = result.numpy()
        return class_names[class_idx], confidence_scores
    except Exception as e:
        return "Error", str(e)

# Function to create a custom progress bar
def custom_progress_bar(confidence, colors):
    bar_html = "<div style=\"border: 1px solid #ddd; border-radius: 5px; overflow: hidden; width: 100%; font-size: 14px;\">"
    for i, (conf, color) in enumerate(zip(confidence, colors)):
        percentage = conf * 100
        bar_html += f"<div style=\"width: {percentage:.2f}%; background: {color}; color: white; text-align: center; height: 24px; float: left;\">"
        bar_html += f"{class_names[i]}: {percentage:.2f}%</div>"
    bar_html += "</div>"
    st.sidebar.markdown(bar_html, unsafe_allow_html=True)

# Streamlit UI
st.title("Klasifikasi Buah: Blackberry, Blueberry, Strawberry")

# Upload multiple files in the main page
uploaded_files = st.file_uploader(
    "Unggah Gambar (JPG/PNG/JPEG)", 
    type=["jpg", "png", "jpeg"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.write("### Gambar yang Diunggah:")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"{uploaded_file.name}", use_column_width=True)

# Sidebar for prediction results
if st.sidebar.button("Prediksi Semua"):
    if uploaded_files:
        st.sidebar.write("### Hasil Prediksi")
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Perform prediction
            label, confidence = classify_image(uploaded_file.name)
            if label != "Error":
                # Define colors for the progress bar
                colors = ["#007BFF", "#FF4136", "#2ECC71"]  # Colors for Blackberry, Blueberry, Strawberry

                # Display prediction results
                st.sidebar.write(f"**Nama File:** {uploaded_file.name}")
                st.sidebar.write(f"**Prediksi:** {label}")
                st.sidebar.write("**Confidence Scores:**")
                for i, class_name in enumerate(class_names):
                    st.sidebar.write(f"- {class_name}: {confidence[i] * 100:.2f}%")

                # Display custom progress bar
                custom_progress_bar(confidence, colors)
                st.sidebar.write("---")
            else:
                st.sidebar.error(f"Kesalahan memproses gambar {uploaded_file.name}: {confidence}")
    else:
        st.sidebar.error("Silakan unggah setidaknya satu gambar untuk diprediksi.")

# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf

# Loading the Model
model = load_model('Model/plant_disease_model.h5')
model.summary()

# Name of Classes
CLASS_NAMES = [
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___healthy',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Soyabean_Septoria_Brown_Spot',
    'Soyabean_Vein Necrosis',
    'Soybean___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___healthy',
    'Tomato___Late_blight'
]

# Setting Title of App
st.title("Plant Disease Identification using Leaf Images")
st.markdown("Upload an image of the plant leaf")

# Uploading the plant image
plant_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
submit = st.button('Predict Disease')

# On predict button click
if submit:
    if plant_image is not None:
        try:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            # Displaying the image
            st.image(opencv_image, channels="BGR")
            st.write(f"Original Image Shape: {opencv_image.shape}")

            # Resizing the image
            opencv_image = cv2.resize(opencv_image, (128, 128))

            # Normalize the image
            opencv_image = opencv_image.astype('float32') / 255.0

            # Reshape image
            opencv_image = np.expand_dims(opencv_image, axis=0)  # shape: (1, 128, 128, 3)

            # Make Prediction
            Y_pred = model.predict(opencv_image)
            predicted_class = CLASS_NAMES[np.argmax(Y_pred)]

            # Display result
            st.title(f"This is a {predicted_class.replace('___', ' with ')} leaf")

        except Exception as e:
            st.error(f"Error processing image: {e}")
    else:
        st.warning("Please upload an image file to predict.")

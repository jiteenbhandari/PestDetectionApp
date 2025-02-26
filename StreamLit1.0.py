import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the TensorFlow model
model = tf.keras.models.load_model('EfficientNetB7.keras')

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Resize the image to the size expected by your model
    image = image.resize((600, 600))  # Change this to your model's input size
    image_array = np.array(image)
    # image_array = image_array / 255.0  # Normalize the image to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Streamlit app
st.title("TensorFlow Model Prediction")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(processed_image)
    print("[Predictions] :", predictions)
    predicted_class = np.argmax(predictions, axis=1)  # Adjust based on your output
    print("[predicted_class] :", predicted_class)
    # Display the prediction result
    st.write(f'Prediction: {predicted_class[0]}')  # Modify this based on your class mapping
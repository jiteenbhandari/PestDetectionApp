
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# Define the class names mapping
class_names = [
    'rice leaf roller',
    'rice leaf caterpillar',
    'paddy stem maggot',
    'asiatic rice borer',
    'yellow rice borer',
    'rice gall midge',
    'Rice Stemfly',
    'brown plant hopper',
    'white backed plant hopper',
    'small brown plant hopper',
    'rice water weevil',
    'rice leafhopper',
    'grain spreader thrips',
    'rice shell pest',
    'grub',
    'mole cricket',
    'wireworm',
    'white margined moth',
    'black cutworm',
    'large cutworm',
    'yellow cutworm',
    'red spider',
    'corn borer',
    'army worm',
    'aphids',
    'Potosiabre vitarsis',
    'peach borer',
    'english grain aphid',
    'green bug',
    'bird cherry-oataphid',
    'wheat blossom midge',
    'penthaleus major',
    'longlegged spider mite',
    'wheat phloeothrips',
    'wheat sawfly',
    'cerodonta denticornis'
]

def load_model(crop_type):
    if crop_type == "Rice":
        return tf.keras.models.load_model('Resnet_Rice.keras')  # Change to your actual model file
    elif crop_type == "Wheat":
        return tf.keras.models.load_model('Resnet_Wheat.keras')  # Change to your actual model file
    elif crop_type == "Maize":
        return tf.keras.models.load_model('Resnet_Maize.keras')  # Change to your actual model file
    else:
        return None

# Function to make the image square by adding padding and then resize to 224x224
def load_and_preprocess_image(image, target_size=(224, 224)):
    # Convert image to RGB (in case it's grayscale or has an alpha channel)
    image = image.convert("RGB")
    
    # Get original dimensions
    width, height = image.size
    
    # Calculate padding
    if width > height:
        padding = (0, (width - height) // 2)  # (left, top)
    else:
        padding = ((height - width) // 2, 0)  # (left, top)
    
    # Create a new square image with a white background
    new_size = max(width, height)
    square_image = Image.new("RGB", (new_size, new_size), (255, 255, 255))
    square_image.paste(image, padding)

    # Resize to target size
    square_image = square_image.resize(target_size)
    
    # Convert the image to an array
    img_array = img_to_array(square_image)
    # Add an extra dimension to the array (for batch size)
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the image
    img_array = img_array / 255.0  # Normalize to [0, 1]
    
    return img_array

# Function to predict the class of an image
def predict_image(model, processed_image):
    # Make predictions
    predictions = model.predict(processed_image)
    # Get the class index of the predicted class
    predicted_class_index = np.argmax(predictions, axis=-1)[0]
    return predicted_class_index

# Streamlit app
st.title("Crop Pest Classification")

# Dropdown for crop selection
options = ["Select a crop", "Rice", "Wheat", "Maize"]
selected_option = st.selectbox("Choose Crop Type", options)

# Load the model based on the selected crop type
model = load_model(selected_option)

if model is not None:
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = load_and_preprocess_image(image)

        # Make predictions
        predicted_class_index = predict_image(model, processed_image)

        # Get the corresponding class name
        predicted_class_name = class_names[predicted_class_index]

        # Display the prediction result
        st.write(f'Prediction Class Index: {predicted_class_index}')  # Display the index
        st.write(f'Predicted Class: {predicted_class_name}')  # Display the mapped class name
else:
    st.write("Please select a crop type to load the corresponding model.")

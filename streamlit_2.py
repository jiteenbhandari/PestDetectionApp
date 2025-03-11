
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

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


def load_and_preprocess_image(img, target_size=(224, 224)):
    # Load the image
    # img = load_img(img, target_size=target_size)
    # Convert the image to an array
    img_array = img_to_array(img)
    # Add an extra dimension to the array (for batch size)
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the image
   
    return img_array

# Function to predict the class of an image
def predict_image(model, img):
    # Preprocess the image
    processed_image = load_and_preprocess_image(img)
    # Make predictions
    predictions = model.predict(processed_image)
    # Get the class index of the predicted class
    print(predictions)
    predicted_class_index = np.argmax(predictions, axis=-1)[0]
    print(predicted_class_index)
    return predicted_class_index



# Streamlit app
st.title("Crop Pest Classification")

# Dropdown for crop selection023

options = ["Select a crop", "Rice", "Wheat", "Maize"]
selected_option = st.selectbox("Choose Crop Type", options)

  
    
if selected_option == "Rice":
        temp = 0
elif selected_option == "Wheat":
        temp = 27
elif selected_option == "Maize":
        temp = 14


# Load the model based on the selected crop type
model = load_model(selected_option)

if model is not None:
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        image = image.resize((224,224))
        st.image(image, caption='Uploaded Image', use_column_width=True)


        # Make predictions
        predicted_class_index = predict_image(model, image)
        print("blablabla ", predicted_class_index)
        predicted_class_index = predicted_class_index + temp

        if predicted_class_index < 6 and predicted_class_index > 1:
             predicted_class_index += 8

        if predicted_class_index < 14 and predicted_class_index > 5:
             predicted_class_index = predicted_class_index - 4


        # Get the corresponding class name
        predicted_class_name = class_names[predicted_class_index]

        # Display the prediction result
        st.write(f'Prediction Class Index: {predicted_class_index}')  # Display the index
        st.write(f'Predicted Class: {predicted_class_name}')  # Display the mapped class name
else:
    st.write("Please select a crop type to load the corresponding model.")

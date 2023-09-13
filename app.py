# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from PIL import Image
# from tensorflow.keras.applications.densenet import preprocess_input
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import os

# def preprocess_image(image_path):
#     # Load the image with target size (150, 150)
#     img = image.load_img(image_path, target_size=(150, 150))
#     # Convert the image to an array
#     img_array = image.img_to_array(img)
#     # Expand dimensions to match the model input shape (None, 150, 150, 3)
#     img_array = np.expand_dims(img_array, axis=0)
#     # Preprocess the image using the preprocess_input function
#     img_array = preprocess_input(img_array)
#     return img_array


# # Load the pre-trained model
# model = load_model('models/pneumonia_model.h5')

# # Streamlit app
# st.title("Pneumonia Classification App")

# # File upload widget
# uploaded_file = st.file_uploader("Upload an X-ray Image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     st.image(uploaded_file, caption="Uploaded Image", width=500)

#     # Preprocess the uploaded image
#     img = preprocess_image(uploaded_file)

#     # Make predictions using your loaded model
#     predictions = model.predict(img)

#     # Define class labels
#     class_labels = ["Normal", "Bacterial", "Viral"]

#     # Get the predicted class index
#     predicted_class_index = np.argmax(predictions)

#     # Get the predicted class label
#     predicted_class_label = class_labels[predicted_class_index]

#     # Display the prediction result
#     st.header(f"Prediction: {predicted_class_label}")




import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Load the pre-trained model
model = load_model('models/pneumonia_model.h5')

# Streamlit app
st.title("Pneumonia Classification App")
st.write("Upload an X-ray image to classify it as Normal, Bacterial, or Viral Pneumonia.")

# File upload widget
uploaded_file = st.file_uploader("Choose an X-ray Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the uploaded image
    img = preprocess_image(uploaded_file)

    # Make predictions using your loaded model
    predictions = model.predict(img)

    # Define class labels
    class_labels = ["Normal", "Bacterial", "Viral"]

    # Get the predicted class index and label
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]

    # Display the prediction result in a styled container
    prediction_container = st.empty()
    with st.spinner('Analyzing...'):
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("")
        prediction_container.header(f"Prediction: {predicted_class_label}")

    # Add style to the prediction container
    prediction_container.markdown(
        f'<div style="background-color: #f5f5f5; padding: 10px; border-radius: 10px; text-align: center;">'
        f'<h3 style="color: #333;">Prediction: {predicted_class_label}</h3>'
        f'</div>', unsafe_allow_html=True)

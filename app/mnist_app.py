import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model("model/mnist_model.h5")

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray, (28, 28))
    scaled_image = resized_image / 255.0  # Scale pixel values to [0, 1]
    reshaped_image = np.reshape(scaled_image, (1, 28, 28))
    return reshaped_image

def add_sidebar():
    st.sidebar.header("Upload Image")

    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    return uploaded_file

def add_predictions(image, model):
    input_image = preprocess_image(image)
    prediction = model.predict(input_image)
    predicted_label = np.argmax(prediction)

    st.subheader("Prediction")
    st.write(f"The predicted digit is: **{predicted_label}**")
    
    # Display the original image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Display probabilities for each class
    st.write("Prediction probabilities for each digit:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"Digit {i}: {prob * 100:.2f}%")
    
    return predicted_label

def main():
    # Set up page
    st.set_page_config(
        page_title="MNIST Digit Classifier",
        page_icon=":1234:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .digit-prediction {
        font-size: 36px;
        font-weight: bold;
        color: #009688;
    }
    </style>
    """, unsafe_allow_html=True)

    model = load_model()

    uploaded_file = add_sidebar()

    if uploaded_file is not None:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, 1)  # Decode the image as BGR format
        
        # Show predictions
        predicted_label = add_predictions(image, model)

        # Display confusion matrix if needed
        st.write("This app can assist with digit recognition. It's a demo of digit classification using a deep learning model.")

if __name__ == "__main__":
    main()

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Title of the app
st.title("Sign Language Detection")

# Class mapping for letters
class_labels = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z',
}

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image for your Keras model (CNN input)
    def preprocess_image(img):
        img = img.resize((64, 64))  # Resize image to match model input size
        img = np.array(img)  # Convert image to numpy array
        img = img / 255.0  # Normalize pixel values between 0 and 1
        img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 64, 64, 3)
        return img

    img_tensor = preprocess_image(img)

    # Load your .h5 Keras model
    model = tf.keras.models.load_model('/workspaces/sign-language-detection-system/asl_model.h5')

    # Print model input shape for verification
    st.write(f"Model Input Shape: {model.input_shape}")

    # Make prediction
    try:
        prediction = model.predict(img_tensor)

        # Get the class with the highest probability
        predicted_class_index = np.argmax(prediction, axis=1)[0]

        # Map the predicted class index to the corresponding letter
        predicted_letter = class_labels[predicted_class_index]

        # Display the result
        st.write(f"Predicted Letter: {predicted_letter}")
    except ValueError as e:
        st.error(f"Prediction Error: {e}")

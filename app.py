import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

st.set_page_config(
        page_title="X-ray Fracture Detector",
        page_icon="🦴",
        layout="centered"
    )

# --- Configuration ---
# Trying an explicit relative path to confirm the file is in the root directory
MODEL_PATH = 'model.h5'
TARGET_SIZE = (224, 224)              # Match the input size your model was trained on
CLASS_NAMES = ['fracture', 'normal']  # Must match the order of your model's output classes

# --- Function to load the Keras model using Streamlit caching ---
# @st.cache_resource is used to load the model only once, speeding up the app significantly.
@st.cache_resource
def load_and_cache_model():
    """Loads the pre-trained Keras model."""
    try:
        # Keras 3 (which uses .keras format) handles model loading, but load_model 
        # is still the correct function to call.
        model = tf.keras.models.load_model(MODEL_PATH, compile=False) 
        return model
    except Exception as e:
        # Re-raise the error to ensure the Streamlit app shows the failure clearly
        raise e 

# --- Function to preprocess the uploaded image and predict ---
def predict_image_class(model, uploaded_file):
    """
    Preprocesses the image file and makes a prediction using the Keras model.
    """
    # Load and resize the image
    img = Image.open(uploaded_file).convert('RGB') 
    img = img.resize(TARGET_SIZE)

    # Convert to array and normalize
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    # Predict
    try:
        predictions = model.predict(img_array)
        # Assuming the model outputs logits or probabilities that need softmax if more than 2 classes,
        # but since we have 2 classes, softmax will give probabilities.
        score = tf.nn.softmax(predictions[0]) 
        
        # Get the result
        predicted_class_index = np.argmax(score)
        confidence = np.max(score) * 100
        predicted_class_name = CLASS_NAMES[predicted_class_index]

        return predicted_class_name, confidence, img
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, img

# --- Streamlit UI and Logic ---
# Initialize the model once
try:
    model = load_and_cache_model()
except Exception as e:
    # If the model load failed, show a simple error message and stop.
    st.error(f"Error loading the model: {e}")
    st.info("Please ensure the 'model.h5' file is uploaded and accessible in the root directory.")
    model = None


if model:
    st.title("🦴 X-ray Bone Fracture Detection")
    st.markdown(
        """
        Upload an X-ray image to check for signs of a bone fracture. 
        The model classifies the image as either **'fracture'** or **'normal'**.
        """
    )
    st.divider()

    # File Uploader Widget
    uploaded_file = st.file_uploader(
        "Choose an X-ray Image (JPG, JPEG, PNG)",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        # Display a spinner while processing
        with st.spinner('Analyzing X-ray...'):
            predicted_class, confidence, display_img = predict_image_class(model, uploaded_file)
        
        # Display the uploaded image
        st.image(display_img, caption='Uploaded X-ray Image', use_column_width=True)
        st.divider()

        if predicted_class:
            if predicted_class == 'fracture':
                st.markdown(
                    f"### ⚠️ **Prediction: {predicted_class.upper()}**", 
                    unsafe_allow_html=True
                )
                st.warning(f"Confidence: **{confidence:.2f}%**")
                st.markdown("🚨 **Recommendation:** *Please seek immediate medical consultation with a qualified professional.*")
            else:
                st.markdown(
                    f"### ✅ **Prediction: {predicted_class.upper()}**", 
                    unsafe_allow_html=True
                )
                st.success(f"Confidence: **{confidence:.2f}%**")
                st.markdown("👍 *The model did not detect a fracture.*")
            
    else:
        st.info("Upload an image to start the diagnosis.")

# --- Footer ---
st.markdown("""
<style>
.stApp {
    background-color: #1E1E1E;
}
</style>
""", unsafe_allow_html=True)
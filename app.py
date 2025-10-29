import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# --- CONFIGURATION (Match your model's settings) ---
MODEL_PATH = 'models/imageclassifier.h5'
IMG_SIZE = (256, 256)
APP_TITLE = "Deep CNN Emotion Classifier 🧠"

# --- HELPER FUNCTIONS ---

# Use Streamlit's cache to load the model only once
@st.cache_resource
def get_model():
    # Placeholder for model loading
    # NOTE: You MUST have your model saved at this path: models/imageclassifier.h5
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    # Resize and normalize as done in your notebook
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def predict_image(model, processed_image):
    prediction = model.predict(processed_image)
    yhat_prob = prediction[0][0] * 100 # Assuming 'Sad' is the positive class (1)

    if yhat_prob > 50:
        label = 'Sad'
        confidence = yhat_prob
        icon = '😞'
    else:
        label = 'Happy'
        confidence = 100 - yhat_prob
        icon = '😊'
        
    return label, confidence, icon

# --- STREAMLIT APP LAYOUT ---

# 1. Page Configuration
st.set_page_config(
    page_title=APP_TITLE,
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# 2. Header and Description
st.markdown(f"<h1 style='text-align: center; color: #333366;'>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
This is a custom-built **Deep Convolutional Neural Network (CNN)** for binary image classification.
Upload an image and see how the model, trained from scratch on a small dataset, predicts one of two emotions: **Happy** or **Sad**.
""")

# Load the model
model = get_model()
if model is None:
    st.stop() # Stop execution if the model failed to load

# 3. File Uploader and Main Content
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "**Upload an Image for Classification**", 
        type=["jpg", "jpeg", "png"],
        help="The image should ideally contain a clear face."
    )

with col2:
    if uploaded_file is None:
        st.info("Waiting for image upload...")
    else:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Image Uploaded', use_column_width=True)
        st.markdown("")
        
        # --- CLASSIFICATION LOGIC ---
        if st.button('Analyze Emotion', use_container_width=True, type="primary"):
            with st.spinner('Running inference on the Deep CNN...'):
                processed_image = preprocess_image(image)
                label, confidence, icon = predict_image(model, processed_image)
                
                st.markdown("### Model Prediction:")
                
                # Use a beautiful colored metric to display the result
                if label == 'Happy':
                    st.metric(label=f"Predicted Emotion {icon}", value=label, delta=f"{confidence:.2f}% Confidence", delta_color="normal")
                else:
                    # Use "inverse" delta color for Sad to look more alarming/serious
                    st.metric(label=f"Predicted Emotion {icon}", value=label, delta=f"{confidence:.2f}% Confidence", delta_color="inverse")
                    
# 4. Footer/Project Details
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center;'>
        **Project Details:** | **Framework:** TensorFlow/Keras | **Deployment:** Streamlit | **Model Size:** {:,} Parameters
    </div>
    """.format(model.count_params()), 
    unsafe_allow_html=True
)
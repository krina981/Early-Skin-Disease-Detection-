import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from keras import backend as K
from PIL import Image
import plotly.express as px

# Define paths
MODELSPATH = 'C:/Users/91932/Desktop/SYDS/RD/jaymataji/model.h5'
DATAPATH = 'C:/Users/91932/Desktop/SYDS/RD/jaymataji/ISIC_0000002.jpg'

# Render Header
def render_header():
    st.write("""
        <p align="center"> 
            <H1> Skin Cancer Analyzer </H1>
        </p>
    """, unsafe_allow_html=True)

# Cache image loading
@st.cache_data
def load_mekd():
    img = Image.open(DATAPATH)
    return img

# Function to preprocess the image
def data_gen(x):
    img = np.asarray(Image.open(x).resize((100, 75)))
    x_test = img.astype(np.float32)
    x_test_mean = np.mean(x_test)
    x_test_std = np.std(x_test)
    x_test = (x_test - x_test_mean) / x_test_std
    x_validate = x_test.reshape(1, 75, 100, 3)
    return x_validate

# Cache model loading
@st.cache_resource
def load_models():
    return load_model(MODELSPATH)

# Prediction function (Remove @st.cache_data)
def predict(x_test, model):
    Y_pred = model.predict(x_test)
    ynew = np.round(Y_pred * 100, 2)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    return ynew[0].tolist(), Y_pred_classes

# Function to display predictions
def display_prediction(y_new):
    result = pd.DataFrame({'Probability': y_new}, index=np.arange(7))
    result = result.reset_index()
    result.columns = ['Classes', 'Probability']
    lesion_type_dict = {
        2: 'Benign keratosis-like lesions',
        4: 'Melanocytic nevi',
        3: 'Dermatofibroma',
        5: 'Melanoma',
        6: 'Vascular lesions',
        1: 'Basal cell carcinoma',
        0: 'Actinic keratoses'
    }
    result["Classes"] = result["Classes"].map(lesion_type_dict)
    return result

# Streamlit App
def main():
    st.sidebar.header('Skin Cancer Analyzer')
    st.sidebar.subheader('Choose a page to proceed:')
    page = st.sidebar.selectbox("", ["Sample Data", "Upload Your Image"])

    if page == "Sample Data":
        st.header("Sample Data Prediction for Skin Cancer")
        st.markdown("""
        **Now, this is probably why you came here. Let's get you some Predictions**  
        You need to choose Sample Data.
        """)

        sample_options = ['Sample Data I']
        sample_chosen = st.multiselect('Choose Sample Data', sample_options)

        if len(sample_chosen) > 1:
            st.error('Please select only one Sample Data')
        elif len(sample_chosen) == 1:
            st.success("You have selected Sample Data")
            if st.checkbox('Show Sample Data'):
                st.info("Displaying Sample Data")
                image = load_mekd()
                st.image(image, caption='Sample Data', use_column_width=True)

                st.subheader("Choose Training Algorithm!")
                if st.checkbox('Keras'):
                    model = load_models()
                    st.success("Keras Model Loaded!")
                    
                    if st.checkbox('Show Prediction Probability on Sample Data'):
                        x_test = data_gen(DATAPATH)
                        y_new, Y_pred_classes = predict(x_test, model)
                        result = display_prediction(y_new)
                        st.write(result)

                        if st.checkbox('Display Probability Graph'):
                            fig = px.bar(result, x="Classes", y="Probability", color='Classes')
                            st.plotly_chart(fig, use_container_width=True)

    if page == "Upload Your Image":
        st.header("Upload Your Image")

        file_path = st.file_uploader('Upload an image', type=['png', 'jpg'])

        if file_path is not None:
            image = Image.open(file_path)
            x_test = data_gen(file_path)
            img_array = np.array(image)

            st.success('File Upload Successful!')
            
            if st.checkbox('Show Uploaded Image'):
                st.info("Displaying Uploaded Image")
                st.image(img_array, caption='Uploaded Image', use_column_width=True)

                st.subheader("Choose Training Algorithm!")
                if st.checkbox('Keras'):
                    model = load_models()
                    st.success("Keras Model Loaded!")

                    if st.checkbox('Show Prediction Probability for Uploaded Image'):
                        y_new, Y_pred_classes = predict(x_test, model)
                        result = display_prediction(y_new)
                        st.write(result)

                        if st.checkbox('Display Probability Graph'):
                            fig = px.bar(result, x="Classes", y="Probability", color='Classes')
                            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

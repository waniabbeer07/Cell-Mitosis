import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained models
anomaly_model = joblib.load('/content/anomaly_detection_model.pkl')
mitosis_model = joblib.load('/content/mitosis_stage_prediction_model.pkl')

# Load the scaler used for standardizing the features during training
scaler = StandardScaler()

# Define the features for prediction
features = ['mean_intensity', 'circularity', 'aspect_ratio']

# Function to make predictions for anomaly detection and mitosis stage
def predict_anomaly_and_mitosis(mean_intensity, circularity, aspect_ratio):
    # Create a DataFrame for input data
    input_data = pd.DataFrame([[mean_intensity, circularity, aspect_ratio]], columns=features)
    
    # Standardize the features using the same scaler used during training
    input_data_scaled = scaler.fit_transform(input_data)

    # Predict anomaly status (normal/abnormal)
    anomaly_prediction = anomaly_model.predict(input_data_scaled)
    anomaly_status = "Normal" if anomaly_prediction == 1 else "Abnormal"

    # Predict mitosis stage
    mitosis_prediction = mitosis_model.predict(input_data_scaled)
    mitosis_stage_mapping = {0: 'Prophase/Metaphase', 1: 'Telophase', 2: 'Anaphase'}
    mitosis_stage = mitosis_stage_mapping[mitosis_prediction[0]]

    return anomaly_status, mitosis_stage

# Streamlit app layout
st.title('Cell Anomaly Detection and Mitosis Stage Prediction')

st.write("""
    This application allows you to predict cell anomaly status (Normal/Abnormal) and mitosis stages (Prophase/Metaphase, 
    Telophase, Anaphase) based on cell features such as mean intensity, circularity, and aspect ratio.
""")

# Input fields for the user to enter the cell features
mean_intensity = st.number_input('Enter Mean Intensity:', min_value=0.0, max_value=100.0, step=0.1)
circularity = st.number_input('Enter Circularity:', min_value=0.0, max_value=1.0, step=0.01)
aspect_ratio = st.number_input('Enter Aspect Ratio:', min_value=0.0, max_value=10.0, step=0.1)

# Button to make predictions
if st.button('Predict'):
    anomaly_status, mitosis_stage = predict_anomaly_and_mitosis(mean_intensity, circularity, aspect_ratio)

    # Display results
    st.subheader('Prediction Results:')
    st.write(f"Anomaly Status: {anomaly_status}")
    st.write(f"Mitosis Stage: {mitosis_stage}")


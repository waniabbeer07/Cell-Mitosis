import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import streamlit as st

# Function to calculate thresholds for each feature
def calculate_thresholds(df, features, quantile=0.9):
    thresholds = {}
    
    for feature in features:
        # Calculate the threshold for each feature based on the quantile
        threshold = df[feature].quantile(quantile)
        
        # Store the thresholds
        thresholds[feature] = threshold
        
        # Visualize the feature distributions
        plt.figure(figsize=(8, 6))
        plt.hist(df[feature], bins=20, alpha=0.5, label=f'{feature} Distribution')
        plt.axvline(threshold, color='blue', linestyle='dashed', linewidth=2, label=f'Threshold ({threshold:.2f})')
        plt.title(f"Feature: {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.legend()
        
        # Streamlit to display the plot
        st.pyplot(plt)
    
    return thresholds

# Function to classify input data based on thresholds
def classify_data(input_data, normal_thresholds, abnormal_thresholds, features):
    # Check for each feature if it's above the normal threshold and below the abnormal threshold
    for feature in features:
        normal_threshold = normal_thresholds[feature]
        abnormal_threshold = abnormal_thresholds[feature]
        
        if input_data[feature] > normal_threshold and input_data[feature] < abnormal_threshold:
            return 'normal'
    
    return 'abnormal'

# Streamlit interface
st.title("Cell Classification App")
st.write("Upload your normal and abnormal data CSV files, and the model will classify the data.")

# File upload for normal and abnormal data
normal_file = st.file_uploader("Upload Normal Data CSV", type="csv")
abnormal_file = st.file_uploader("Upload Abnormal Data CSV", type="csv")

# File path for saved model (if available)
model_file_path = 'cell_classification_model.pkl'

# Load the model if it exists
try:
    with open(model_file_path, 'rb') as file:
        model = pickle.load(file)
    normal_thresholds = model['normal_thresholds']
    abnormal_thresholds = model['abnormal_thresholds']
    st.write("Model loaded from saved file.")
except FileNotFoundError:
    st.write("No saved model found. Please upload normal and abnormal data to generate thresholds.")

if normal_file and abnormal_file:
    # Read the uploaded files
    normal_df = pd.read_csv(normal_file)
    abnormal_df = pd.read_csv(abnormal_file)
    
    # Define the feature columns to calculate thresholds for
    features = ['mean_intensity', 'aspect_ratio', 'circularity']

    # Calculate thresholds for each feature (For normal data and abnormal data)
    normal_thresholds = calculate_thresholds(normal_df, features)
    abnormal_thresholds = calculate_thresholds(abnormal_df, features)

    # Save the model (thresholds) as a pickle file
    model = {
        'normal_thresholds': normal_thresholds,
        'abnormal_thresholds': abnormal_thresholds
    }
    
    with open(model_file_path, 'wb') as file:
        pickle.dump(model, file)

    st.write(f"Model saved as '{model_file_path}'")

# Feature selection and input data for classification
st.sidebar.header("Input Data for Classification")

# Let user input any features
selected_features = st.sidebar.multiselect("Select Features", ['mean_intensity', 'aspect_ratio', 'circularity'])

# Create input fields based on selected features
input_data = {}
for feature in selected_features:
    input_data[feature] = st.sidebar.number_input(f"Enter {feature}", min_value=0.0, step=0.01)

# Classify the input data if the model is loaded
if 'normal_thresholds' in locals() and 'abnormal_thresholds' in locals() and selected_features:
    classification_result = classify_data(input_data, normal_thresholds, abnormal_thresholds, selected_features)
    st.write(f"The input data is classified as: **{classification_result}**")
else:
    st.write("Please upload the normal and abnormal data to generate a model for classification and select at least one feature to classify.")

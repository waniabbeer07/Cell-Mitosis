import pandas as pd
import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

# Function to calculate thresholds for each feature
def calculate_thresholds(df, features, quantile=0.9):
    thresholds = {}
    
    for feature in features:
        # Calculate the threshold for each feature based on the quantile (90th percentile)
        threshold = df[feature].quantile(quantile)
        thresholds[feature] = threshold
        
        # Visualize the feature distributions
        plt.figure(figsize=(8, 6))
        plt.hist(df[feature], bins=20, alpha=0.5, label=f'{feature} Distribution')
        plt.axvline(threshold, color='blue', linestyle='dashed', linewidth=2, label=f'Threshold ({threshold:.2f})')
        plt.title(f"Feature: {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        st.pyplot(plt)  # Display plot in Streamlit
    
    return thresholds

# Function to classify input data based on normal and abnormal thresholds
def classify_data(input_data, normal_thresholds, abnormal_thresholds, features):
    classification = 'normal'  # Default classification
    
    for feature in features:
        # Get the threshold values for the feature
        normal_threshold = normal_thresholds[feature]
        abnormal_threshold = abnormal_thresholds[feature]
        
        feature_value = input_data[feature]
        
        # If feature exceeds the abnormal threshold, classify as abnormal
        if feature_value >= abnormal_threshold:
            classification = 'abnormal'
            break
        # If feature is below the normal threshold, classify as normal
        elif feature_value <= normal_threshold:
            classification = 'normal'
            break
        else:
            classification = 'abnormal'
    
    return classification

# Streamlit app
def main():
    st.title("Cell Classification App")
    
    # Upload normal and abnormal CSV files
    normal_file = st.file_uploader("Upload normal data CSV", type=["csv"])
    abnormal_file = st.file_uploader("Upload abnormal data CSV", type=["csv"])
    
    # Option to upload a pre-trained classification model .pkl file
    model_file = st.file_uploader("Upload your pre-trained model file", type=["pkl"])
    
    if normal_file is not None and abnormal_file is not None:
        # Load CSV data
        normal_df = pd.read_csv(normal_file)
        abnormal_df = pd.read_csv(abnormal_file)
        
        # Define the feature columns to calculate thresholds for
        features = ['mean_intensity', 'aspect_ratio', 'circularity']
        
        # Calculate thresholds for each feature (For normal data and abnormal data)
        normal_thresholds = calculate_thresholds(normal_df, features)
        abnormal_thresholds = calculate_thresholds(abnormal_df, features)
        
        # If no model file is uploaded, save the calculated thresholds as a model
        if model_file is None:
            model = {
                'normal_thresholds': normal_thresholds,
                'abnormal_thresholds': abnormal_thresholds
            }
            
            # Save model as a pickle file in memory
            model_pickle = BytesIO()
            pickle.dump(model, model_pickle)
            model_pickle.seek(0)
            
            # Provide a download button for the user to download the model
            st.download_button(
                label="Download Generated Classification Model",
                data=model_pickle,
                file_name="cell_classification_model.pkl",
                mime="application/octet-stream"
            )
            st.write("Model generated. You can download the model file.")
        else:
            # If a model file is uploaded, load it
            model_pickle = model_file.read()
            loaded_model = pickle.loads(model_pickle)
            
            normal_thresholds = loaded_model['normal_thresholds']
            abnormal_thresholds = loaded_model['abnormal_thresholds']
            
            st.write("Model loaded from the uploaded file.")
        
        # Input fields for classification
        st.subheader("Enter feature values to classify the data:")
        mean_intensity = st.number_input("Mean Intensity", value=0.0)
        aspect_ratio = st.number_input("Aspect Ratio", value=0.0)
        circularity = st.number_input("Circularity", value=0.0)
        
        input_data = {
            'mean_intensity': mean_intensity,
            'aspect_ratio': aspect_ratio,
            'circularity': circularity
        }
        
        # Classify the input data
        if st.button("Classify"):
            classification_result = classify_data(input_data, normal_thresholds, abnormal_thresholds, features)
            st.write(f"The input data is classified as: {classification_result}")

# Run the app
if __name__ == "__main__":
    main()

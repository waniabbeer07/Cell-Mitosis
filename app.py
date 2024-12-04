# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from model_train import CombinedKDEModel  # Import the class definition

# Streamlit app layout
st.title('Cell Classification App')
st.write('This app classifies cells as Normal or Abnormal based on their features.')

# File upload section
st.subheader('Upload Your Normal and Abnormal Cell CSV Files')

# Upload Normal Cells Data
normal_file = st.file_uploader("Upload the CSV file for Normal Cells", type=['csv'])
# Upload Abnormal Cells Data
abnormal_file = st.file_uploader("Upload the CSV file for Abnormal Cells", type=['csv'])

if normal_file is not None and abnormal_file is not None:
    # Load the datasets
    normal_df = pd.read_csv(normal_file)
    abnormal_df = pd.read_csv(abnormal_file)
    
    # Adding labels
    normal_df['label'] = 0  # Normal cells are labeled as 0
    abnormal_df['label'] = 1  # Abnormal cells are labeled as 1
    
    # Combine datasets
    data = pd.concat([normal_df, abnormal_df])
    
    st.write(f"Normal cells dataset: {normal_df.shape[0]} rows and {normal_df.shape[1]} columns")
    st.write(f"Abnormal cells dataset: {abnormal_df.shape[0]} rows and {abnormal_df.shape[1]} columns")
    
    # Display first few rows of the combined dataset
    st.write("First few rows of the combined dataset:")
    st.write(data.head())
else:
    st.write("Please upload both the Normal and Abnormal cells datasets in CSV format.")

# File upload for model
st.subheader('Upload Your Trained Model')

model_file = st.file_uploader("Upload your trained model (.pkl)", type=['pkl'])
if model_file is not None:
    model = joblib.load(model_file)
    st.write("Model loaded successfully!")

# Feature input from the user
st.subheader('Enter Cell Features')

mean_intensity = st.number_input('Mean Intensity', min_value=0.0, max_value=200.0, value=130.0)
circularity = st.number_input('Circularity', min_value=0.0, max_value=1.0, value=0.7)
aspect_ratio = st.number_input('Aspect Ratio', min_value=0.0, max_value=2.0, value=1.0)

# Validate if the files and model are loaded
if model_file is not None and normal_file is not None and abnormal_file is not None:
    # Checking if required columns exist in the dataset
    if all(col in data.columns for col in ['mean_intensity', 'circularity', 'aspect_ratio']):
        # Predict button to classify the cell
        if st.button('Classify'):
            # Prepare the input data for prediction
            input_data = pd.DataFrame({
                'mean_intensity': [mean_intensity],
                'circularity': [circularity],
                'aspect_ratio': [aspect_ratio]
            })
            
            # Standardize the input data
            scaler = StandardScaler()
            input_data_scaled = scaler.fit_transform(input_data)
            
            # Make prediction using the loaded model
            prediction = model.predict(input_data_scaled)
            
            # Map the prediction to a label
            if prediction == 0:
                st.write(f'The cell is classified as: Normal')
            else:
                st.write(f'The cell is classified as: Abnormal')
    else:
        st.write("The dataset must contain the columns: 'mean_intensity', 'circularity', and 'aspect_ratio'.")

# Optional: Display feature distributions using seaborn
st.subheader('Feature Distributions')

if normal_file is not None and abnormal_file is not None:
    # Plot feature distributions for the uploaded data
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot Mean Intensity Distribution
    sns.histplot(data[data['label'] == 0]['mean_intensity'], color='blue', label='Normal', kde=True, ax=axes[0])
    sns.histplot(data[data['label'] == 1]['mean_intensity'], color='red', label='Abnormal', kde=True, ax=axes[0])
    axes[0].set_title('Mean Intensity Distribution')
    axes[0].set_xlabel('Mean Intensity')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # Plot Circularity Distribution
    sns.histplot(data[data['label'] == 0]['circularity'], color='blue', label='Normal', kde=True, ax=axes[1])
    sns.histplot(data[data['label'] == 1]['circularity'], color='red', label='Abnormal', kde=True, ax=axes[1])
    axes[1].set_title('Circularity Distribution')
    axes[1].set_xlabel('Circularity')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    # Plot Aspect Ratio Distribution
    sns.histplot(data[data['label'] == 0]['aspect_ratio'], color='blue', label='Normal', kde=True, ax=axes[2])
    sns.histplot(data[data['label'] == 1]['aspect_ratio'], color='red', label='Abnormal', kde=True, ax=axes[2])
    axes[2].set_title('Aspect Ratio Distribution')
    axes[2].set_xlabel('Aspect Ratio')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()

    plt.tight_layout()
    st.pyplot(fig)

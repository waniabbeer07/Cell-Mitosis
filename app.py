import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from model_train import KDEModel  # Import the KDEModel class

# Streamlit app layout
st.title('Cell Classification App')
st.write('This app classifies cells as Normal or Abnormal based on their features.')

# File upload section
st.subheader('Upload Your Normal and Abnormal Cell CSV Files')

# Upload Normal Cells Data
normal_file = st.file_uploader("Upload the CSV file for Normal Cells", type=['csv'])
# Upload Abnormal Cells Data
abnormal_file = st.file_uploader("Upload the CSV file for Abnormal Cells", type=['csv'])

if normal_file is not None:
    # Load the normal cells dataset
    normal_df = pd.read_csv(normal_file)
    st.write(f"Normal cells dataset: {normal_df.shape[0]} rows and {normal_df.shape[1]} columns")
    st.write("First few rows of the normal cells dataset:")
    st.write(normal_df.head())

if abnormal_file is not None:
    # Load the abnormal cells dataset
    abnormal_df = pd.read_csv(abnormal_file)
    st.write(f"Abnormal cells dataset: {abnormal_df.shape[0]} rows and {abnormal_df.shape[1]} columns")
    st.write("First few rows of the abnormal cells dataset:")
    st.write(abnormal_df.head())

# Upload Model files
st.subheader('Upload Your Trained Models')

normal_model_file = st.file_uploader("Upload your trained Normal Cells model (.pkl)", type=['pkl'])
abnormal_model_file = st.file_uploader("Upload your trained Abnormal Cells model (.pkl)", type=['pkl'])

if normal_model_file is not None:
    normal_model = joblib.load(normal_model_file)
    st.write("Normal cells model loaded successfully!")

if abnormal_model_file is not None:
    abnormal_model = joblib.load(abnormal_model_file)
    st.write("Abnormal cells model loaded successfully!")

# Feature input from the user
st.subheader('Enter Cell Features')

mean_intensity = st.number_input('Mean Intensity', min_value=0.0, max_value=200.0, value=130.0)
circularity = st.number_input('Circularity', min_value=0.0, max_value=1.0, value=0.7)
aspect_ratio = st.number_input('Aspect Ratio', min_value=0.0, max_value=2.0, value=1.0)

# Check if models and datasets are loaded
if normal_model_file is not None and abnormal_model_file is not None:
    # Validate if the input data has the required columns
    if 'mean_intensity' in normal_df.columns and 'circularity' in normal_df.columns and 'aspect_ratio' in normal_df.columns:
        
        # Classify normal cell feature
        if st.button('Classify Normal Cell'):
            input_data = pd.DataFrame({
                'mean_intensity': [mean_intensity],
                'circularity': [circularity],
                'aspect_ratio': [aspect_ratio]
            })
            
            # Standardize the input data
            scaler = StandardScaler()
            input_data_scaled = scaler.fit_transform(input_data)
            
            # Make prediction using the normal cell model
            normal_prediction = normal_model.classify(input_data_scaled)
            
            # Display result
            st.write(f'The cell is classified as: {"Normal" if normal_prediction == 0 else "Abnormal"}')
        
        # Classify abnormal cell feature
        if st.button('Classify Abnormal Cell'):
            input_data = pd.DataFrame({
                'mean_intensity': [mean_intensity],
                'circularity': [circularity],
                'aspect_ratio': [aspect_ratio]
            })
            
            # Standardize the input data
            scaler = StandardScaler()
            input_data_scaled = scaler.fit_transform(input_data)
            
            # Make prediction using the abnormal cell model
            abnormal_prediction = abnormal_model.classify(input_data_scaled)
            
            # Display result
            st.write(f'The cell is classified as: {"Normal" if abnormal_prediction == 0 else "Abnormal"}')

# Optional: Display feature distributions using seaborn
st.subheader('Feature Distributions')

if normal_file is not None and abnormal_file is not None:
    # Plot feature distributions for the uploaded data
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot Mean Intensity Distribution
    sns.histplot(normal_df['mean_intensity'], color='blue', label='Normal', kde=True, ax=axes[0])
    sns.histplot(abnormal_df['mean_intensity'], color='red', label='Abnormal', kde=True, ax=axes[0])
    axes[0].set_title('Mean Intensity Distribution')
    axes[0].set_xlabel('Mean Intensity')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # Plot Circularity Distribution
    sns.histplot(normal_df['circularity'], color='blue', label='Normal', kde=True, ax=axes[1])
    sns.histplot(abnormal_df['circularity'], color='red', label='Abnormal', kde=True, ax=axes[1])
    axes[1].set_title('Circularity Distribution')
    axes[1].set_xlabel('Circularity')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    # Plot Aspect Ratio Distribution
    sns.histplot(normal_df['aspect_ratio'], color='blue', label='Normal', kde=True, ax=axes[2])
    sns.histplot(abnormal_df['aspect_ratio'], color='red', label='Abnormal', kde=True, ax=axes[2])
    axes[2].set_title('Aspect Ratio Distribution')
    axes[2].set_xlabel('Aspect Ratio')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()

    plt.tight_layout()
    st.pyplot(fig)

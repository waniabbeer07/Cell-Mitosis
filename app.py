import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app layout
st.title('Cell Classification App')
st.write('This app classifies cells as Normal or Abnormal based on their features.')

# Upload section for the model files
st.subheader('Upload Your Trained Models')

normal_model_file = st.file_uploader("Upload the Trained Normal Cells Model (.pkl)", type=['pkl'])
abnormal_model_file = st.file_uploader("Upload the Trained Abnormal Cells Model (.pkl)", type=['pkl'])

# Check if models are uploaded
if normal_model_file is not None:
    kde_normal = joblib.load(normal_model_file)
    st.write("Normal cells model loaded successfully!")

if abnormal_model_file is not None:
    kde_abnormal = joblib.load(abnormal_model_file)
    st.write("Abnormal cells model loaded successfully!")

# Feature input from the user
st.subheader('Enter Cell Features')

mean_intensity = st.number_input('Mean Intensity', min_value=0.0, max_value=200.0, value=130.0)
circularity = st.number_input('Circularity', min_value=0.0, max_value=1.0, value=0.7)
aspect_ratio = st.number_input('Aspect Ratio', min_value=0.0, max_value=2.0, value=1.0)

# Check if models are loaded and perform classification
if normal_model_file is not None and abnormal_model_file is not None:
    # Prepare the input data for classification
    input_data = np.array([[mean_intensity, circularity, aspect_ratio]])
    
    # Standardize the input data
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    
    # Calculate log-likelihoods for both classes
    log_likelihood_normal = kde_normal.score_samples(input_data_scaled)
    log_likelihood_abnormal = kde_abnormal.score_samples(input_data_scaled)
    
    # Classify based on the higher log-likelihood
    if log_likelihood_normal > log_likelihood_abnormal:
        st.write("The cell is classified as: Normal")
    else:
        st.write("The cell is classified as: Abnormal")

# Optional: Display feature distributions using seaborn
st.subheader('Feature Distributions')

# Display distribution for normal and abnormal cells if CSVs are uploaded
normal_file = st.file_uploader("Upload the Normal Cells CSV File", type=['csv'])
abnormal_file = st.file_uploader("Upload the Abnormal Cells CSV File", type=['csv'])

if normal_file is not None and abnormal_file is not None:
    normal_df = pd.read_csv(normal_file)
    abnormal_df = pd.read_csv(abnormal_file)
    
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

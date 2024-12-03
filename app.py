
import streamlit as st
import pandas as pd
import joblib

# Streamlit UI
st.title("Telophase Cell Classifier")
st.write("Upload the trained model file (pkl), normal cells data, and abnormal cells data.")

# File upload for the trained model
model_file = st.file_uploader("Upload the trained model (.pkl)", type=["pkl"])

# File upload for the normal and abnormal cell CSV files
normal_file = st.file_uploader("Upload normal cell data (CSV)", type=["csv"])
abnormal_file = st.file_uploader("Upload abnormal cell data (CSV)", type=["csv"])

# Process the uploaded model file if available
model = None
if model_file is not None:
    model = joblib.load(model_file)
    st.write("Model loaded successfully!")

# Process the uploaded CSV files for normal and abnormal cells
normal_telophase = None
abnormal_telophase = None
if normal_file is not None:
    normal_telophase = pd.read_csv(normal_file)
    st.write("Normal cell data loaded successfully!")

if abnormal_file is not None:
    abnormal_telophase = pd.read_csv(abnormal_file)
    st.write("Abnormal cell data loaded successfully!")

# Ensure that both normal and abnormal files are uploaded
if normal_telophase is not None and abnormal_telophase is not None and model is not None:
    # Combine the data for training if needed
    normal_telophase['label'] = 0  # label 0 for normal
    abnormal_telophase['label'] = 1  # label 1 for abnormal
    data = pd.concat([normal_telophase, abnormal_telophase])

    # Define features
    features = ['mean_intensity', 'area', 'perimeter', 'circularity']

    # Streamlit input fields for user to input cell features for classification
    st.write("Enter the feature values of the Telophase cell to classify it as Normal or Abnormal.")

    mean_intensity = st.number_input("Mean Intensity", min_value=0.0, max_value=150.0, step=0.1)
    area = st.number_input("Area", min_value=0.0, max_value=10000.0, step=0.1)
    perimeter = st.number_input("Perimeter", min_value=0.0, max_value=1000.0, step=0.1)
    circularity = st.number_input("Circularity", min_value=0.0, max_value=1.0, step=0.01)

    # When the user submits the input, classify the cell
    if st.button("Classify"):
        feature_values = {
            'mean_intensity': mean_intensity,
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity
        }

        # Classify using the trained model
        features_input = pd.DataFrame([feature_values])  # Convert input to DataFrame for prediction
        classification = model.predict(features_input)

        if classification[0] == 0:
            st.write("Based on the provided features, the cell is classified as: **Normal**")
        else:
            st.write("Based on the provided features, the cell is classified as: **Abnormal**")
else:
    st.warning("Please upload the model and both the normal and abnormal cell data files.")

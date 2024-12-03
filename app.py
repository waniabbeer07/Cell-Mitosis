
import streamlit as st
import pandas as pd

# Read the uploaded CSV files
normal_telophase = pd.read_csv('normal_cells.csv')
abnormal_telophase = pd.read_csv('abnormal_cells.csv')

# Calculate the threshold values (you can adjust the method as needed)
threshold_values = {
    'mean_intensity': normal_telophase['mean_intensity'].median(),
    'area': normal_telophase['area'].median(),
    'perimeter': normal_telophase['perimeter'].median(),
    'circularity': normal_telophase['circularity'].median()
}

def classify_cell(feature_values, thresholds):
    for feature, value in feature_values.items():
        if value > thresholds[feature]:
            return 'Abnormal'
    return 'Normal'

# Streamlit UI
st.title("Telophase Cell Classifier")
st.write("Enter the feature values of the Telophase cells to classify them as Normal or Abnormal.")

mean_intensity = st.number_input("Mean Intensity", min_value=0.0, max_value=100.0, step=0.1)
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

    classification = classify_cell(feature_values, threshold_values)

    st.write(f"Based on the provided features, the cell is classified as: {classification}")

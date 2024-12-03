import streamlit as st
import pandas as pd
import joblib

# Load the trained classifier
classifier = joblib.load('cell_classifier.pkl')

# Streamlit UI
st.title("Telophase Cell Classifier")
st.write("Enter the feature values of the Telophase cells to classify them as Normal or Abnormal.")

# Input fields for user to input feature values
mean_intensity = st.number_input("Mean Intensity", min_value=0.0, max_value=100.0, step=0.1)
area = st.number_input("Area", min_value=0.0, max_value=10000.0, step=0.1)
perimeter = st.number_input("Perimeter", min_value=0.0, max_value=1000.0, step=0.1)
circularity = st.number_input("Circularity", min_value=0.0, max_value=1.0, step=0.01)

# When the user submits the input, classify the cell
if st.button("Classify"):
    # Prepare the feature vector
    feature_values = {
        'mean_intensity': mean_intensity,
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity
    }
    input_data = pd.DataFrame([feature_values])

    # Make prediction using the trained classifier
    prediction = classifier.predict(input_data)

    # Display the result
    if prediction == 1:
        st.write("The cell is classified as: Abnormal")
    else:
        st.write("The cell is classified as: Normal")

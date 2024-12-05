import streamlit as st
import pandas as pd
from cell_classifier import CellClassifier

# Set up Streamlit app
st.title("Cell Classification App")
st.write("This app classifies cells as Normal or Abnormal based on their features.")

# Load data
st.header("Step 1: Upload Datasets")
normal_file = st.file_uploader("Upload Normal Cells CSV", type="csv")
abnormal_file = st.file_uploader("Upload Abnormal Cells CSV", type="csv")

if normal_file and abnormal_file:
    normal_df = pd.read_csv(normal_file)
    abnormal_df = pd.read_csv(abnormal_file)
    st.success("Data loaded successfully!")

    # Initialize the classifier
    features = ['mean_intensity', 'circularity', 'aspect_ratio']
    classifier = CellClassifier(features)

    # Prepare and train the model
    classifier.load_and_prepare_data(normal_df, abnormal_df)
    st.header("Step 2: Train the Model")
    if st.button("Train Model"):
        accuracy, report = classifier.train_model()
        st.write("Balanced Accuracy:", accuracy)
        st.text("Classification Report:")
        st.text(report)

    # Predict new data
    st.header("Step 3: Make Predictions")
    mean_intensity = st.number_input("Mean Intensity", min_value=0.0, value=0.0)
    circularity = st.number_input("Circularity", min_value=0.0, value=0.0)
    aspect_ratio = st.number_input("Aspect Ratio", min_value=0.0, value=0.0)

    if st.button("Classify"):
        try:
            result = classifier.predict([mean_intensity, circularity, aspect_ratio])
            st.write(f"Prediction: {result}")
        except ValueError as e:
            st.error(str(e))


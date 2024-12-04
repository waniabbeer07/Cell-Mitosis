# streamlit_classifier.py
import streamlit as st
import pickle

# Load thresholds from the model file
@st.cache
def load_model(model_path):
    with open(model_path, 'rb') as model_file:
        return pickle.load(model_file)

# Classify user input based on thresholds
def classify_single_input(input_values, thresholds):
    results = {}
    for feature, value in input_values.items():
        if feature not in thresholds:
            results[feature] = "unknown"  # Handle missing feature thresholds
            continue
        t = thresholds[feature]
        if t['normal_min'] <= value <= t['normal_max']:
            results[feature] = "normal"
        elif t['abnormal_min'] <= value <= t['abnormal_max']:
            results[feature] = "abnormal"
        else:
            results[feature] = "unknown"
    return results

# Streamlit UI
st.title("Real-Time Threshold-Based Classifier")

# Step 1: Load Threshold Model
st.header("Step 1: Load Pre-trained Threshold Model")
model_file = st.file_uploader("Upload Threshold Model (threshold_model.pkl)", type=["pkl"])

if model_file:
    thresholds = load_model(model_file)
    st.success("Model loaded successfully!")

    # Step 2: User Input for Classification
    st.header("Step 2: Enter Feature Values for Real-Time Classification")
    st.subheader("Input values for each feature:")

    # Input fields for each feature
    mean_intensity = st.number_input("Mean Intensity", min_value=0.0, step=0.1)
    area = st.number_input("Area", min_value=0.0, step=0.1)
    perimeter = st.number_input("Perimeter", min_value=0.0, step=0.1)
    circularity = st.number_input("Circularity", min_value=0.0, step=0.01)

    # Classify button
    if st.button("Classify"):
        input_values = {
            'mean_intensity': mean_intensity,
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity
        }
        classification_results = classify_single_input(input_values, thresholds)
        
        st.subheader("Classification Results:")
        for feature, classification in classification_results.items():
            st.write(f"**{feature}**: {classification}")

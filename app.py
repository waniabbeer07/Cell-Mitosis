import streamlit as st
import pandas as pd
import pickle
import tempfile

def generate_model(normal_csv_path, abnormal_csv_path, output_model_path='threshold_model.pkl'):
    # Load the datasets
    normal_df = pd.read_csv(normal_csv_path)
    abnormal_df = pd.read_csv(abnormal_csv_path)

    # Define the features we are interested in
    features = ['mean_intensity', 'aspect_ratio', 'circularity']

    # Calculate min and max thresholds for each feature
    thresholds = {}
    for feature in features:
        thresholds[feature] = {
            'normal_min': normal_df[feature].min(),
            'normal_max': normal_df[feature].max(),
            'abnormal_min': abnormal_df[feature].min(),
            'abnormal_max': abnormal_df[feature].max()
        }

    # Save the thresholds to a file
    with open(output_model_path, 'wb') as model_file:
        pickle.dump(thresholds, model_file)

    st.success(f"Threshold model saved to {output_model_path}")
    return thresholds

def classify_input(input_values, thresholds):
    classification = {}
    for feature, value in input_values.items():
        if thresholds[feature]['normal_min'] <= value <= thresholds[feature]['normal_max']:
            classification[feature] = 'normal'
        elif thresholds[feature]['abnormal_min'] <= value <= thresholds[feature]['abnormal_max']:
            classification[feature] = 'abnormal'
        else:
            classification[feature] = 'out of bounds'
    return classification

# Streamlit UI
st.title("Threshold Classifier for Cell Features")

# Option to upload or generate threshold model
thresholds = None
threshold_model_file = st.file_uploader("Upload your threshold model (Pickle file)", type=["pkl"])

if threshold_model_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_model:
        tmp_model.write(threshold_model_file.getvalue())
        threshold_model_path = tmp_model.name
    try:
        with open(threshold_model_path, 'rb') as model_file:
            thresholds = pickle.load(model_file)
        st.success("Threshold model loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

# Always allow the user to upload CSV files for model generation
normal_file = st.file_uploader("Upload the normal dataset (CSV)", type=["csv"])
abnormal_file = st.file_uploader("Upload the abnormal dataset (CSV)", type=["csv"])

if normal_file and abnormal_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_normal, tempfile.NamedTemporaryFile(delete=False) as tmp_abnormal:
        tmp_normal.write(normal_file.getvalue())
        tmp_abnormal.write(abnormal_file.getvalue())
        normal_temp_path = tmp_normal.name
        abnormal_temp_path = tmp_abnormal.name

    if not thresholds:
        thresholds = generate_model(normal_temp_path, abnormal_temp_path)
        st.write("Generated thresholds:")
        st.write(thresholds)

# User input for classification
st.subheader("Enter feature values to classify:")
input_values = {
    'mean_intensity': st.number_input("Enter value for mean intensity", value=0.0),
    'aspect_ratio': st.number_input("Enter value for aspect ratio", value=0.0),
    'circularity': st.number_input("Enter value for circularity", value=0.0)
}

if st.button("Classify"):
    if thresholds:
        classification = classify_input(input_values, thresholds)
        st.write("Classification Results:")
        for feature, result in classification.items():
            st.write(f"{feature}: {result}")
    else:
        st.error("Threshold model is not available. Please upload or generate one.")

import streamlit as st
import pandas as pd
import pickle
import tempfile

def generate_model(normal_csv_path, abnormal_csv_path, output_model_path='threshold_model.pkl'):
    # Load the datasets
    normal_df = pd.read_csv(normal_csv_path)
    abnormal_df = pd.read_csv(abnormal_csv_path)

    # Define the features we are interested in
    features = ['min_intensity', 'aspect_ratio', 'circularity']

    # Calculate min and max thresholds for each feature
    thresholds = {}
    for feature in features:
        normal_min = normal_df[feature].min()
        normal_max = normal_df[feature].max()
        abnormal_min = abnormal_df[feature].min()
        abnormal_max = abnormal_df[feature].max()

        thresholds[feature] = {
            'normal_min': normal_min,
            'normal_max': normal_max,
            'abnormal_min': abnormal_min,
            'abnormal_max': abnormal_max
        }

    # Save the thresholds to a file
    with open(output_model_path, 'wb') as model_file:
        pickle.dump(thresholds, model_file)

    st.success(f"Threshold model saved to {output_model_path}")
    return thresholds

def classify_input(input_values, thresholds):
    classification = {}
    for feature, value in input_values.items():
        # Check if the value is within the normal range
        if thresholds[feature]['normal_min'] <= value <= thresholds[feature]['normal_max']:
            classification[feature] = 'normal'
        elif thresholds[feature]['abnormal_min'] <= value <= thresholds[feature]['abnormal_max']:
            classification[feature] = 'abnormal'
        else:
            classification[feature] = 'out of bounds'
    return classification

# Streamlit UI
st.title("Threshold Classifier for Cell Features")

# Option to upload a threshold model
threshold_model_file = st.file_uploader("Upload your own threshold model (Pickle file)", type=["pkl"])

if threshold_model_file is not None:
    # Load the user's custom threshold model
    with tempfile.NamedTemporaryFile(delete=False) as tmp_model:
        tmp_model.write(threshold_model_file.getvalue())
        threshold_model_path = tmp_model.name

    with open(threshold_model_path, 'rb') as model_file:
        thresholds = pickle.load(model_file)

    st.success("Threshold model loaded successfully.")
    st.write("Using the uploaded threshold model.")

else:
    # File uploaders for normal and abnormal datasets if model is not uploaded
    normal_file = st.file_uploader("Upload the normal dataset (CSV)", type=["csv"])
    abnormal_file = st.file_uploader("Upload the abnormal dataset (CSV)", type=["csv"])

    if normal_file is not None and abnormal_file is not None:
        # Save the uploaded files to temporary locations
        with tempfile.NamedTemporaryFile(delete=False) as tmp_normal, tempfile.NamedTemporaryFile(delete=False) as tmp_abnormal:
            tmp_normal.write(normal_file.getvalue())
            tmp_abnormal.write(abnormal_file.getvalue())
            normal_temp_path = tmp_normal.name
            abnormal_temp_path = tmp_abnormal.name

        # Generate the model with uploaded files
        thresholds = generate_model(normal_temp_path, abnormal_temp_path)

        # Show a preview of the thresholds
        st.write("Generated thresholds:")
        st.write(thresholds)

# Get user input for classification of the three features
st.subheader("Enter feature values to classify:")

input_values = {}
input_values['min_intensity'] = st.number_input("Enter value for min intensity", value=0.0)
input_values['aspect_ratio'] = st.number_input("Enter value for aspect ratio", value=0.0)
input_values['circularity'] = st.number_input("Enter value for circularity", value=0.0)

if st.button("Classify"):
    if thresholds:
        classification = classify_input(input_values, thresholds)
        st.write("Classification Results:")
        for feature, result in classification.items():
            st.write(f"{feature}: {result}")

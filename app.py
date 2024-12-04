import streamlit as st
import pandas as pd
import pickle
import tempfile

def generate_model(normal_csv_path, abnormal_csv_path, output_model_path='threshold_model.pkl'):
    # Load the datasets
    normal_df = pd.read_csv(normal_csv_path)
    abnormal_df = pd.read_csv(abnormal_csv_path)

    # Calculate thresholds for each feature
    thresholds = {}
    features = normal_df.columns
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

# Streamlit UI
st.title("Threshold Model Generator")

# File uploaders for normal and abnormal datasets
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

else:
    st.warning("Please upload both normal and abnormal datasets to generate the model.")

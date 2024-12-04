import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils import resample
import pickle
import tempfile

# Function to generate and save Naive Bayes model and scaler
def generate_and_save_model(normal_csv_path, abnormal_csv_path, output_model_path='threshold_model.pkl', output_scaler_path='scaler_model.pkl'):
    # Read the normal and abnormal CSV files
    normal_df = pd.read_csv(normal_csv_path)
    abnormal_df = pd.read_csv(abnormal_csv_path)

    # Define the features we are interested in
    features = ['mean_intensity', 'aspect_ratio', 'circularity']

    # Add a label column before concatenating: 1 for normal, 0 for abnormal
    normal_df['label'] = 1  # 1 for normal
    abnormal_df['label'] = 0  # 0 for abnormal

    # Combine the normal and abnormal datasets
    combined_df = pd.concat([normal_df, abnormal_df])

    # Separate majority and minority classes
    normal_df = combined_df[combined_df['label'] == 1]
    abnormal_df = combined_df[combined_df['label'] == 0]

    # Upsample the minority class (abnormal)
    abnormal_upsampled = resample(abnormal_df, 
                                  replace=True,     # Sample with replacement
                                  n_samples=len(normal_df),  # Match the number of normal samples
                                  random_state=42)  # Reproducibility

    # Combine the upsampled data with the normal data
    combined_upsampled = pd.concat([normal_df, abnormal_upsampled])

    # Prepare the feature matrix and target vector
    X = combined_upsampled[features]
    y = combined_upsampled['label']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Naive Bayes model
    model = GaussianNB()
    model.fit(X_scaled, y)

    # Save the trained model and scaler to pickle files
    with open(output_model_path, 'wb') as model_file, open(output_scaler_path, 'wb') as scaler_file:
        pickle.dump(model, model_file)
        pickle.dump(scaler, scaler_file)

    st.success(f"Model and scaler saved to {output_model_path} and {output_scaler_path}")

# Function to classify input
def classify_input(input_values, model, scaler, threshold=0.5):
    # Define features for classification
    features = ['mean_intensity', 'aspect_ratio', 'circularity']

    # Convert the input dictionary to a DataFrame
    input_df = pd.DataFrame([input_values], columns=features)
    
    # Scale the input
    input_scaled = scaler.transform(input_df)
    
    # Get predicted probabilities for each class (0 and 1)
    probabilities = model.predict_proba(input_scaled)
    
    # Check if the predicted probability for class 1 (abnormal) is above the threshold
    prediction = 1 if probabilities[0][1] > threshold else 0
    return 'abnormal' if prediction == 1 else 'normal'

# Streamlit UI
st.title("Naive Bayes Classifier for Cell Features")

# File upload option for CSV files
normal_file = st.file_uploader("Upload the normal dataset (CSV)", type=["csv"])
abnormal_file = st.file_uploader("Upload the abnormal dataset (CSV)", type=["csv"])

# Option to upload a previously trained model and scaler
threshold_model_file = st.file_uploader("Upload your trained Naive Bayes model (Pickle file)", type=["pkl"])
scaler_model_file = st.file_uploader("Upload your scaler model (Pickle file)", type=["pkl"])

# Load model and scaler if files are provided
model = None
scaler = None
if threshold_model_file is not None and scaler_model_file is not None:
    try:
        # Load the Naive Bayes model
        with tempfile.NamedTemporaryFile(delete=False) as tmp_model:
            tmp_model.write(threshold_model_file.getvalue())
            model_path = tmp_model.name
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        
        # Load the scaler model
        with tempfile.NamedTemporaryFile(delete=False) as tmp_scaler:
            tmp_scaler.write(scaler_model_file.getvalue())
            scaler_path = tmp_scaler.name
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        
        st.success("Model and scaler loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load model or scaler: {e}")

# If CSV files are uploaded, generate and save the model
if normal_file and abnormal_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_normal, tempfile.NamedTemporaryFile(delete=False) as tmp_abnormal:
        tmp_normal.write(normal_file.getvalue())
        tmp_abnormal.write(abnormal_file.getvalue())
        normal_temp_path = tmp_normal.name
        abnormal_temp_path = tmp_abnormal.name

    # Generate and save the model if none is loaded
    if not model:
        generate_and_save_model(normal_temp_path, abnormal_temp_path)

# User input for classification
st.subheader("Enter feature values to classify:")

input_values = {
    'mean_intensity': st.number_input("Enter value for mean intensity", value=0.0),
    'aspect_ratio': st.number_input("Enter value for aspect ratio", value=0.0),
    'circularity': st.number_input("Enter value for circularity", value=0.0)
}

# Perform classification when button is clicked
if st.button("Classify"):
    if model and scaler:
        classification = classify_input(input_values, model, scaler)
        st.write("Classification Result:")
        st.write(f"The cell is classified as: {classification}")
    else:
        st.error("Please upload or generate a model first.")

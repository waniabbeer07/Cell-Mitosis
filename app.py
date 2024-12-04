import streamlit as st
import pandas as pd
import pickle
import tempfile
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Function to generate Naive Bayes model with hyperparameter tuning
def generate_model(normal_csv_path, abnormal_csv_path, output_model_path='naive_bayes_model.pkl'):
    # Load the datasets
    normal_df = pd.read_csv(normal_csv_path)
    abnormal_df = pd.read_csv(abnormal_csv_path)

    # Define the features we are interested in
    features = ['mean_intensity', 'aspect_ratio', 'circularity']

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

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Naive Bayes model
    model = GaussianNB()

    # Hyperparameter tuning using GridSearchCV (for Naive Bayes, adjusting var_smoothing)
    param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='precision')
    grid_search.fit(X_train, y_train)

    # Best hyperparameters
    st.write(f"Best parameters found: {grid_search.best_params_}")

    # Train model with best parameters
    best_model = grid_search.best_estimator_

    # Fit the model on the entire dataset
    best_model.fit(X_train, y_train)

    # Save the trained model to a pickle file
    with open(output_model_path, 'wb') as model_file:
        pickle.dump(best_model, model_file)

    st.success(f"Naive Bayes model saved to {output_model_path}")

    return best_model, scaler

# Function to classify input using the trained model
def classify_input(input_values, model, scaler, threshold=0.5):
    # Convert the input dictionary to a DataFrame
    input_df = pd.DataFrame([input_values], columns=['mean_intensity', 'aspect_ratio', 'circularity'])
    
    # Scale the input
    input_scaled = scaler.transform(input_df)
    
    # Get predicted probabilities for each class (0 and 1)
    probabilities = model.predict_proba(input_scaled)
    
    # Check if the predicted probability for class 1 (abnormal) is above the threshold
    prediction = 1 if probabilities[0][1] > threshold else 0
    return 'abnormal' if prediction == 1 else 'normal'

# Streamlit UI
st.title("Naive Bayes Classifier for Cell Features")

# Option to upload or generate threshold model
model = None
scaler = None
threshold_model_file = st.file_uploader("Upload your Naive Bayes model (Pickle file)", type=["pkl"])

if threshold_model_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_model:
        tmp_model.write(threshold_model_file.getvalue())
        model_path = tmp_model.name
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        st.success("Model loaded successfully.")
        # Load the scaler file if available
        scaler_file = st.file_uploader("Upload your scaler model (Pickle file)", type=["pkl"])
        if scaler_file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_scaler:
                tmp_scaler.write(scaler_file.getvalue())
                scaler_path = tmp_scaler.name
            try:
                with open(scaler_path, 'rb') as scaler_file:
                    scaler = pickle.load(scaler_file)
                st.success("Scaler loaded successfully.")
            except Exception as e:
                st.error(f"Failed to load scaler: {e}")
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

    if not model:
        model, scaler = generate_model(normal_temp_path, abnormal_temp_path)
        st.write("Generated model and scaler.")
    
# User input for classification
st.subheader("Enter feature values to classify:")
input_values = {
    'mean_intensity': st.number_input("Enter value for mean intensity", value=0.0),
    'aspect_ratio': st.number_input("Enter value for aspect ratio", value=0.0),
    'circularity': st.number_input("Enter value for circularity", value=0.0)
}

if st.button("Classify"):
    if model and scaler:
        classification = classify_input(input_values, model, scaler)
        st.write("Classification Results:")
        st.write(f"Classification: {classification}")
    else:
        st.error("Model and scaler are not available. Please upload or generate one.")

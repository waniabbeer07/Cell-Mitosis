import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier

# Streamlit UI for uploading files and inputs
st.title("Cell Classification App")

# Upload the trained model
model_file = st.file_uploader("Upload your trained model (.pkl)", type=["pkl"])
if model_file is not None:
    # Load the model
    clf = pickle.load(model_file)
    st.success("Model loaded successfully!")

# Upload normal and abnormal CSV data files
normal_file = st.file_uploader("Upload normal cell data (CSV)", type=["csv"])
abnormal_file = st.file_uploader("Upload abnormal cell data (CSV)", type=["csv"])

if normal_file is not None and abnormal_file is not None:
    # Load the CSV files into pandas DataFrames
    normal_df = pd.read_csv(normal_file)
    abnormal_df = pd.read_csv(abnormal_file)

    # Display a preview of the data
    st.subheader("Normal Cell Data Preview")
    st.write(normal_df.head())
    st.subheader("Abnormal Cell Data Preview")
    st.write(abnormal_df.head())

    # Combine the datasets for training
    normal_df['label'] = 0  # Normal cells are labeled as 0
    abnormal_df['label'] = 1  # Abnormal cells are labeled as 1
    data = pd.concat([normal_df, abnormal_df])

    # Feature columns
    features = ['mean_intensity', 'circularity', 'aspect_ratio']
    X = data[features]
    y = data['label']

    # Train the classifier (if not provided)
    if model_file is None:
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        clf.fit(X, y)
        st.success("Model trained on the uploaded data!")

# Button to request user input and classify data
if st.button("Enter Features for Classification"):
    # Input feature values from the user
    st.subheader("Input features for classification")
    mean_intensity = st.number_input("Mean Intensity", value=131.0, min_value=0.0)
    circularity = st.number_input("Circularity", value=0.5, min_value=0.0, max_value=1.0)
    aspect_ratio = st.number_input("Aspect Ratio", value=1.0, min_value=0.0)

    # Predict the classification based on user input
    if st.button("Classify"):
        input_data = pd.DataFrame({
            'mean_intensity': [mean_intensity],
            'circularity': [circularity],
            'aspect_ratio': [aspect_ratio]
        })

        # Predict using the trained model
        prediction = clf.predict(input_data)
        if prediction == 0:
            st.write("The cell is classified as: **Normal**")
        else:
            st.write("The cell is classified as: **Abnormal**")

# Optional: Save the trained model for future use
if st.button("Save Model"):
    if model_file is not None:
        with open('trained_model.pkl', 'wb') as file:
            pickle.dump(clf, file)
        st.success("Model saved successfully as 'trained_model.pkl'")
    else:
        st.warning("Please upload or train a model before saving.")

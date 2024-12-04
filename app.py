import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier

# Streamlit app structure
st.title("Cell Classification App")

# Upload Model
model_file = st.file_uploader("Upload your trained model", type=["pkl"])
if model_file is not None:
    model = pickle.load(model_file)
    st.success("Model successfully loaded!")

# Upload Normal and Abnormal CSVs
normal_csv = st.file_uploader("Upload Normal CSV", type=["csv"])
abnormal_csv = st.file_uploader("Upload Abnormal CSV", type=["csv"])

# Handle feature inputs
if 'mean_intensity' not in st.session_state:
    st.session_state.mean_intensity = 131.0
if 'circularity' not in st.session_state:
    st.session_state.circularity = 0.5
if 'aspect_ratio' not in st.session_state:
    st.session_state.aspect_ratio = 1.0

st.subheader("Input features for classification")
st.session_state.mean_intensity = st.number_input("Mean Intensity", value=st.session_state.mean_intensity, min_value=0.0)
st.session_state.circularity = st.number_input("Circularity", value=st.session_state.circularity, min_value=0.0, max_value=1.0)
st.session_state.aspect_ratio = st.number_input("Aspect Ratio", value=st.session_state.aspect_ratio, min_value=0.0)

# Show the uploaded CSVs if they are loaded
if normal_csv is not None:
    normal_df = pd.read_csv(normal_csv)
    st.write("Normal Data Preview:", normal_df.head())

if abnormal_csv is not None:
    abnormal_df = pd.read_csv(abnormal_csv)
    st.write("Abnormal Data Preview:", abnormal_df.head())

# Predict when inputs and model are available
if model_file is not None and normal_csv is not None and abnormal_csv is not None:
    # Combine the normal and abnormal data for training
    normal_df['label'] = 0  # Normal cells are labeled as 0
    abnormal_df['label'] = 1  # Abnormal cells are labeled as 1
    data = pd.concat([normal_df, abnormal_df])

    # Feature columns
    features = ['mean_intensity', 'area', 'perimeter', 'circularity', 'aspect_ratio']
    X = data[features]
    y = data['label']

    # Train classifier (for demo purposes; your model would already be trained and loaded)
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    clf.fit(X, y)

    # Predict using the user input
    input_data = {
        'mean_intensity': st.session_state.mean_intensity,
        'circularity': st.session_state.circularity,
        'aspect_ratio': st.session_state.aspect_ratio
    }
    
    new_data = pd.DataFrame([input_data])

    # Predict using the classifier
    prediction = clf.predict(new_data)
    if prediction == 0:
        st.write("Prediction: Normal")
    else:
        st.write("Prediction: Abnormal")

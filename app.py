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

# Handle feature inputs directly with up to 4 significant digits
st.subheader("Input features for classification")
mean_intensity = st.number_input("Mean Intensity", value=131.0, min_value=0.0, format="%.4f")
circularity = st.number_input("Circularity", value=0.5, min_value=0.0, max_value=1.0, format="%.4f")
aspect_ratio = st.number_input("Aspect Ratio", value=1.0, min_value=0.0, format="%.4f")

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
    features = ['mean_intensity','circularity', 'aspect_ratio']
    X = data[features]
    y = data['label']

    # Train classifier (for demo purposes; your model would already be trained and loaded)
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    clf.fit(X, y)

    # Prepare input data for prediction
    input_data = {
        'mean_intensity': mean_intensity,
        'circularity': circularity,
        'aspect_ratio': aspect_ratio
    }
    
    new_data = pd.DataFrame([input_data])

    # Predict using the classifier
    prediction = clf.predict(new_data)
    if prediction == 0:
        st.write("Prediction: Normal")
    else:
        st.write("Prediction: Abnormal")

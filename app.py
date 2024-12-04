import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Function to calculate thresholds for each feature
def calculate_thresholds(df, features, quantile=0.9):
    thresholds = {}
    
    for feature in features:
        # Calculate the threshold for each feature based on the quantile
        threshold = df[feature].quantile(quantile)
        
        # Store the thresholds
        thresholds[feature] = threshold
        
        # Visualize the feature distributions
        plt.figure(figsize=(8, 6))
        plt.hist(df[feature], bins=20, alpha=0.5, label=f'{feature} Distribution')
        plt.axvline(threshold, color='blue', linestyle='dashed', linewidth=2, label=f'Threshold ({threshold:.2f})')
        plt.title(f"Feature: {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
    
    return thresholds

# Function to classify input data based on thresholds
def classify_data(input_data, normal_thresholds, abnormal_thresholds, features):
    # Classify based on threshold conditions
    for feature in features:
        normal_threshold = normal_thresholds[feature]
        abnormal_threshold = abnormal_thresholds[feature]
        
        if input_data[feature] < normal_threshold:
            return 'normal'
        elif input_data[feature] >= abnormal_threshold:
            return 'abnormal'
    
    # If no feature triggers a classification, return 'normal'
    return 'normal'

# Load your CSV data (Normal and Abnormal Data)
normal_df = pd.read_csv("normal_cells.csv")
abnormal_df = pd.read_csv("abnormal_cells.csv")

# Define the feature columns to calculate thresholds for
features = ['mean_intensity', 'aspect_ratio', 'circularity']

# Calculate thresholds for each feature (For normal data and abnormal data)
normal_thresholds = calculate_thresholds(normal_df, features)
abnormal_thresholds = calculate_thresholds(abnormal_df, features)

# Now let's assume we have some input data for classification
input_data = {
    'mean_intensity': 130.94,
    'aspect_ratio': 1.07,
    'circularity': 0.67
}

# Classify the input data
classification_result = classify_data(input_data, normal_thresholds, abnormal_thresholds, features)

# Output the result
print(f"The input data is classified as: {classification_result}")

# Save the thresholds as a pickle file
model = {
    'normal_thresholds': normal_thresholds,
    'abnormal_thresholds': abnormal_thresholds
}

with open('cell_classification_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as 'cell_classification_model.pkl'")

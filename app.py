import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# Load your data
df = pd.read_csv('/content/data.csv')

# Drop the 'frame' column if it exists
df = df.drop(columns=['frame'], errors='ignore')

# Save the modified dataframe as a new CSV file
df.to_csv('/content/new_data.csv', index=False)

# Combine the datasets and add labels (assuming you have 'normal_df' and 'abnormal_df' already)
normal_df = df[df['anomaly_status'] == 1]  # Assuming normal cells are labeled as 1
abnormal_df = df[df['anomaly_status'] == -1]  # Assuming abnormal cells are labeled as -1

normal_df['anomaly_status'] = 1  # Normal cells are labeled as 1
abnormal_df['anomaly_status'] = -1  # Abnormal cells are labeled as -1
data = pd.concat([normal_df, abnormal_df])

# Features to include (excluding 'frame')
features = ['mean_intensity', 'circularity', 'aspect_ratio']
X = data[features]
y = data['anomaly_status']

# Resample the dataset to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Gaussian Naive Bayes model for anomaly detection (Normal/Abnormal classification)
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = nb.predict(X_test_scaled)

# Classification Report for anomaly detection
print("Balanced Classification Report for Anomaly Detection:")
print(classification_report(y_test, y_pred))

# Balanced Accuracy Score for anomaly detection
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy Score for Anomaly Detection: {balanced_accuracy:.4f}")

# Now let's predict the mitosis stages using the same model
# Map the 'mitosis_stage' column to numerical values using the mapping
mitosis_mapping = {'Prophase/Metaphase': 0, 'Telophase': 1, 'Anaphase': 2}

# Assuming 'mitosis_stage' is a column in your data with labels for mitosis stages
df['mitosis_stage_num'] = df['mitosis_stage'].map(mitosis_mapping)

# Drop rows with NaN values in the target variable after mapping
df = df.dropna(subset=['mitosis_stage_num'])

# Features and target for mitosis stage prediction
X_mitosis = df[features]
y_mitosis = df['mitosis_stage_num']

# Split into train-test for mitosis prediction
X_train_mitosis, X_test_mitosis, y_train_mitosis, y_test_mitosis = train_test_split(X_mitosis, y_mitosis, test_size=0.3, random_state=42)

# Train a Gaussian Naive Bayes model for mitosis stage prediction
nb_mitosis = GaussianNB()
nb_mitosis.fit(X_train_mitosis, y_train_mitosis)

# Predict on the test set for mitosis stages
y_pred_mitosis = nb_mitosis.predict(X_test_mitosis)

# Classification Report for mitosis stage prediction
print("\nBalanced Classification Report for Mitosis Stage Prediction:")
print(classification_report(y_test_mitosis, y_pred_mitosis))

# Balanced Accuracy Score for mitosis stage prediction
balanced_accuracy_mitosis = balanced_accuracy_score(y_test_mitosis, y_pred_mitosis)
print(f"Balanced Accuracy Score for Mitosis Stage Prediction: {balanced_accuracy_mitosis:.4f}")

# Now let's add the predicted mitosis stages as a new column in the original dataframe
df['predicted_mitosis_stage'] = nb_mitosis.predict(df[features])

# Map the numeric values back to stage names
mitosis_reverse_mapping = {0: 'Prophase/Metaphase', 1: 'Telophase', 2: 'Anaphase'}
df['predicted_mitosis_stage'] = df['predicted_mitosis_stage'].map(mitosis_reverse_mapping)

# Save the DataFrame with predicted mitosis stages as a new CSV file
df.to_csv('/content/data_with_predicted_mitosis_stages.csv', index=False)

# Optionally print the predicted mitosis stages for the test set
print("\nPredicted Mitosis Stages for the test set:")
for i, stage in enumerate(df['predicted_mitosis_stage']):
    print(f"Sample {i+1}: Mitosis Stage - {stage}")

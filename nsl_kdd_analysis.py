import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load NSL-KDD dataset (adjust file path as needed)
# NSL-KDD usually has files like KDDTrain+.txt and KDDTest+.txt
train_data = pd.read_csv('/archive/nsl-kdd/KDDTrain+.txt', header=None)

# Display basic information
print("Dataset shape:", train_data.shape)
print("\nFirst 5 rows:")
print(train_data.head())

# Check the last column (usually contains attack labels)
print("\nAttack types distribution:")
print(train_data[41].value_counts())
# Add this to your existing file
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Define column names for better understanding
columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
           'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot']
# ... (remaining 33 columns would be listed here)

# Separate features and target
X = train_data.iloc[:, :-2]  # All columns except last 2
y = train_data.iloc[:, 41]   # Attack type column

print("Features shape:", X.shape)
print("Target distribution:")
print(y.value_counts())
# Encode categorical features (columns 1, 2, 3 are typically categorical)
label_encoders = {}
categorical_columns = [1, 2, 3]  # protocol_type, service, flag

for col in categorical_columns:
    le = LabelEncoder()
    X.iloc[:, col] = le.fit_transform(X.iloc[:, col].astype(str))
    label_encoders[col] = le

# Encode target variable (normal vs attack types)
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Data preprocessing completed!")
print("Encoded target classes:", target_encoder.classes_)
print("Number of unique classes:", len(target_encoder.classes_))
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Train a simple Random Forest model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

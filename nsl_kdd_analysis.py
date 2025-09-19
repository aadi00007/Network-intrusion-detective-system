import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load NSL-KDD dataset (adjust file path as needed)
# NSL-KDD usually has files like KDDTrain+.txt and KDDTest+.txt
train_data = pd.read_csv('KDDTrain+.txt', header=None)

# Display basic information
print("Dataset shape:", train_data.shape)
print("\nFirst 5 rows:")
print(train_data.head())

# Check the last column (usually contains attack labels)
print("\nAttack types distribution:")
print(train_data[41].value_counts())

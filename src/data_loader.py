
# ============================================
# MEMBER 1: DATASET LOADING
# Breast Cancer Classification using MLOps
# ============================================

# Import required libraries
import pandas as pd
import os

from sklearn.datasets import load_breast_cancer

# Step 1: Load Wisconsin Breast Cancer Dataset
print("Loading Breast Cancer Dataset...")

data = load_breast_cancer()

# Step 2: Convert features into DataFrame
df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

# Step 3: Add target column
df["target"] = data.target

# Step 4: Create directory
os.makedirs("data/raw", exist_ok=True)

# Step 5: Save dataset as CSV
df.to_csv(
    "data/raw/breast_cancer.csv",
    index=False
)

# Step 6: Display dataset information
print("\nDataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns.tolist())

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nTarget Distribution:")
print(df["target"].value_counts())

print("\nTarget Names:")
print(data.target_names)

print("\nDataset saved at data/raw/breast_cancer.csv")

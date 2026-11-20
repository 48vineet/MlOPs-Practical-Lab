
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# ==========================================
# 1. LOAD THE DATASET
# ==========================================

print("\nLoading Wisconsin Breast Cancer Dataset...")

DATA_PATH = "data/raw/breast_cancer.csv"

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. UNDERSTAND THE DATASET
# ==========================================

print("\nFirst five records:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nStatistical description:")
print(df.describe())


# ==========================================
# 3. CHECK MISSING VALUES
# ==========================================

print("\nChecking missing values:")

missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])

if df.isnull().sum().sum() > 0:
    raise ValueError(
        "Missing values found. Please handle them before splitting."
    )
else:
    print("No missing values found!")


# ==========================================
# 4. CHECK DUPLICATES
# ==========================================

print("\nChecking duplicate records:")

duplicates = df.duplicated().sum()

print("Duplicate records:", duplicates)

if duplicates > 0:
    df = df.drop_duplicates()

    print("Duplicates removed.")

print("Dataset shape after cleaning:", df.shape)


# ==========================================
# 5. SEPARATE FEATURES AND TARGET
# ==========================================

TARGET_COLUMN = "target"

if TARGET_COLUMN not in df.columns:
    raise ValueError("Target column not found in dataset!")

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# ==========================================
# 6. CHECK TARGET DISTRIBUTION
# ==========================================

print("\nTarget class distribution:")

print(y.value_counts().sort_index())

print("\nTarget class meanings:")
print("0 = Malignant")
print("1 = Benign")


# ==========================================
# 7. SPLIT DATA INTO TRAIN AND TEST
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nData splitting completed!")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 8. COMBINE FEATURES AND TARGET
# ==========================================

train_data = X_train.copy()
train_data[TARGET_COLUMN] = y_train

test_data = X_test.copy()
test_data[TARGET_COLUMN] = y_test


# ==========================================
# 9. SAVE PROCESSED DATA
# ==========================================

os.makedirs("data/processed", exist_ok=True)

train_data.to_csv(
    "data/processed/train.csv",
    index=False
)

test_data.to_csv(
    "data/processed/test.csv",
    index=False
)


# ==========================================
# 10. FINAL OUTPUT
# ==========================================

print("\n===================================")
print("DATA PREPROCESSING COMPLETED")
print("===================================")

print("Training dataset:", train_data.shape)
print("Testing dataset:", test_data.shape)

print("\nFiles saved successfully:")
print("1. data/processed/train.csv")
print("2. data/processed/test.csv")

print("\nPreprocessing pipeline executed successfully!")


import pandas as pd

# Load the dataset
df = pd.read_csv("data/heart.csv")

# Basic information
print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

print("\n===== TARGET DISTRIBUTION =====")
if "target" in df.columns:
    print(df["target"].value_counts())
else:
    print("Target column not found. Check the column names above.")

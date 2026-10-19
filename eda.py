
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create folder for charts
os.makedirs("reports", exist_ok=True)

# Load dataset
df = pd.read_csv("data/heart.csv")

print("Original dataset shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())

# Examine dataset without duplicate rows
df_unique = df.drop_duplicates().copy()

print("\nAfter removing exact duplicate rows:")
print("Dataset shape:", df_unique.shape)
print("\nTarget distribution:")
print(df_unique["target"].value_counts())

# Save the deduplicated dataset for later comparison
df_unique.to_csv("data/heart_unique.csv", index=False)

# Set chart style
sns.set_theme(style="whitegrid")

# Chart 1: Target distribution after deduplication
plt.figure(figsize=(7, 5))
sns.countplot(data=df_unique, x="target")
plt.title("Heart Disease Target Distribution")
plt.xlabel("Target (0 = No disease label, 1 = Disease label)")
plt.ylabel("Number of rows")
plt.tight_layout()
plt.savefig("reports/target_distribution.png")
plt.close()

# Chart 2: Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df_unique, x="age", hue="target",
             bins=20, kde=True, element="step")
plt.title("Age Distribution by Target")
plt.tight_layout()
plt.savefig("reports/age_distribution.png")
plt.close()

# Chart 3: Correlation heatmap
plt.figure(figsize=(12, 9))
sns.heatmap(
    df_unique.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("reports/correlation_heatmap.png")
plt.close()

print("\nEDA completed!")
print("Deduplicated data saved to data/heart_unique.csv")
print("Charts saved in the reports folder.")


import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Use the deduplicated dataset for this first experiment
df = pd.read_csv("data/heart_unique.csv")

X = df.drop(columns=["target"])
y = df["target"]

# Keep the same class proportions in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    })

    trained_models[name] = model

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

# Save the model comparison table
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1 Score", ascending=False
)

results_df.to_csv("reports/model_comparison.csv", index=False)

print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))

# Save the best model by test-set F1 for this initial experiment
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

joblib.dump(best_model, "models/heart_model.joblib")

# Save the feature order for the web app
joblib.dump(X.columns.tolist(), "models/feature_columns.joblib")

print(f"\nBest model by test F1: {best_name}")
print("Saved model to models/heart_model.joblib")
print("Saved feature columns to models/feature_columns.joblib")


import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Load deduplicated data
df = pd.read_csv("data/heart_unique.csv")

X = df.drop(columns=["target"])
y = df["target"]

# Five stratified folds
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
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

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

results = []

for name, model in models.items():
    print(f"\nEvaluating {name}...")

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    row = {"Model": name}

    for metric in scoring:
        values = scores[f"test_{metric}"]
        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_std"] = values.std()

    results.append(row)

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="f1_mean",
    ascending=False
)

print("\n===== 5-FOLD CROSS-VALIDATION RESULTS =====")
print(results_df.round(3).to_string(index=False))

results_df.to_csv(
    "reports/cross_validation_results.csv",
    index=False
)

print("\nSaved results to reports/cross_validation_results.csv")

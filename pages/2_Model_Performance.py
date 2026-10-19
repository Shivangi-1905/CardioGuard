
import streamlit as st
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"

st.set_page_config(
    page_title="Model Performance | CardioGuard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Performance")
st.write(
    "Explore the initial test-set results from the "
    "three machine learning models."
)

csv_path = REPORTS_DIR / "model_comparison.csv"

if not csv_path.exists():
    st.error("Model comparison file not found. Train the models first.")
    st.stop()

results = pd.read_csv(csv_path)

# Metrics for the best model in the saved comparison
best = results.sort_values("F1 Score", ascending=False).iloc[0]

col1, col2, col3 = st.columns(3)

col1.metric("Best model by F1", best["Model"])
col2.metric("Accuracy", f'{best["Accuracy"]:.1%}')
col3.metric("ROC-AUC", f'{best["ROC-AUC"]:.3f}')

st.markdown("### Model comparison")
st.dataframe(
    results.style.format({
        "Accuracy": "{:.1%}",
        "Precision": "{:.1%}",
        "Recall": "{:.1%}",
        "F1 Score": "{:.1%}",
        "ROC-AUC": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)

metric = st.selectbox(
    "Choose a metric to compare",
    ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
)

chart_data = results.set_index("Model")[[metric]]
st.bar_chart(chart_data)

st.markdown("### Exploratory data analysis")

for filename, title in [
    ("target_distribution.png", "Target distribution"),
    ("age_distribution.png", "Age distribution"),
    ("correlation_heatmap.png", "Feature correlation heatmap")
]:
    path = REPORTS_DIR / filename
    if path.exists():
        st.markdown(f"#### {title}")
        st.image(str(path), use_container_width=True)

st.warning(
    "These are preliminary results from a small holdout set "
    "using a dataset with substantial duplication. They do not "
    "establish clinical performance or real-world reliability."
)

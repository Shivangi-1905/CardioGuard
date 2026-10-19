import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="CardioGuard | Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

# In this Kaggle file, target 0 = heart disease, target 1 = no disease.
# Verify with: df.groupby("target")[["thalach", "oldpeak", "ca", "exang"]].mean()
# If target 1 has LOWER thalach and HIGHER oldpeak/ca/exang, change this to 1.
DISEASE_CLASS = 0

# Load trained model and feature order
model_path = BASE_DIR / "models" / "heart_model.joblib"
columns_path = BASE_DIR / "models" / "feature_columns.joblib"

@st.cache_resource
def load_model():
    model = joblib.load(model_path)
    columns = joblib.load(columns_path)
    return model, columns

st.title("❤️ CardioGuard")
st.subheader("Heart Disease Prediction")
st.write(
    "Explore a machine learning prediction based on "
    "the health measurements entered below."
)

st.info(
    "Educational project only. This tool is not a medical "
    "device and cannot diagnose heart disease or replace "
    "assessment by a qualified healthcare professional."
)

try:
    model, feature_columns = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

st.markdown("### Enter the health measurements")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age (years)", min_value=1, max_value=120,
            value=50
        )

        sex = st.selectbox(
            "Sex (dataset coding)",
            options=[1, 0],
            format_func=lambda x: {
                0: "0 — Female",
                1: "1 — Male"
            }[x]
        )

        cp = st.selectbox(
            "Chest pain type (cp)",
            options=[0, 1, 2, 3],
            help="Encoded category from the dataset."
        )

        trestbps = st.number_input(
            "Resting blood pressure (mm Hg)",
            min_value=50, max_value=300, value=120
        )

        chol = st.number_input(
            "Serum cholesterol (mg/dL)",
            min_value=50, max_value=700, value=200
        )

        fbs = st.selectbox(
            "Fasting blood sugar above 120 mg/dL (fbs)",
            options=[0, 1],
            format_func=lambda x: {
                0: "0 — No",
                1: "1 — Yes"
            }[x]
        )

        restecg = st.selectbox(
            "Resting ECG result (restecg)",
            options=[0, 1, 2]
        )

    with col2:
        thalach = st.number_input(
            "Maximum heart rate achieved (thalach)",
            min_value=40, max_value=250, value=150
        )

        exang = st.selectbox(
            "Exercise-induced angina (exang)",
            options=[0, 1],
            format_func=lambda x: {
                0: "0 — No",
                1: "1 — Yes"
            }[x]
        )

        oldpeak = st.number_input(
            "Oldpeak (ST depression)",
            min_value=0.0, max_value=10.0,
            value=1.0, step=0.1
        )

        slope = st.selectbox(
            "ST segment slope (slope)",
            options=[0, 1, 2]
        )

        # ca = 4 is a data error in the dataset, so it is removed
        ca = st.selectbox(
            "Major vessels (ca)",
            options=[0, 1, 2, 3]
        )

        # thal = 0 is a data error in the dataset, so it is removed; default = 2
        thal = st.selectbox(
            "Thalassemia category (thal)",
            options=[1, 2, 3],
            index=1,
            format_func=lambda x: {
                1: "1 — Fixed defect",
                2: "2 — Normal",
                3: "3 — Reversible defect"
            }[x]
        )

    submitted = st.form_submit_button(
        "Predict", type="primary", use_container_width=True
    )

if submitted:
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    # Ensure the exact feature order used during training
    input_data = input_data[feature_columns]

    try:
        classes = list(model.classes_)
        proba = model.predict_proba(input_data)[0]
        disease_prob = float(proba[classes.index(DISEASE_CLASS)])

        st.markdown("---")
        st.markdown("### Prediction result")

        st.metric("Model-estimated probability of heart disease", f"{disease_prob:.1%}")
        st.progress(disease_prob)

        if disease_prob >= 0.5:
            st.warning(
                "The model estimates a HIGH likelihood of heart disease "
                "for these inputs. This is not a diagnosis."
            )
        else:
            st.success(
                "The model estimates a LOW likelihood of heart disease "
                "for these inputs. This does not rule out heart disease."
            )

        st.caption(
            "The displayed probability is the model's output, "
            "not a clinically validated estimate of your "
            "personal probability of having heart disease."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
# CardioGuard — Heart Disease Prediction

CardioGuard is a machine learning web application that explores heart disease classification using health-related features from a public Kaggle dataset.

## Features
- Interactive prediction form built with Streamlit
- Logistic Regression, Random Forest, and Gradient Boosting comparison
- Accuracy, precision, recall, F1-score, and ROC-AUC evaluation
- Exploratory data analysis and visualization
- Saved model for local inference

## Tech Stack
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Joblib

## Live Demo 
https://cardioguard-4jpsj57qcuzhayhdiqhu6c.streamlit.app/

## Dataset
Heart Disease Dataset by johnsmith88 on Kaggle:
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

The dataset contains 1,025 rows and 14 columns in the downloaded version used for this project. Exact duplicate removal leaves 302 rows. Both versions are retained for analysis.

## Run Locally

1. Create and activate a Python virtual environment.
2. Install dependencies:

   `pip install -r requirements.txt`

3. Train the models:

   `python train_model.py`

4. Start the web application:

   `python -m streamlit run app.py`

## Important Limitations
This project is for educational and demonstration purposes only. It is not a medical device and must not be used to diagnose, screen, or make treatment decisions. The dataset contains substantial duplication, and the initial evaluation uses a small holdout set. The reported metrics are preliminary and do not demonstrate clinical validity or generalization to real patients.

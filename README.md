# Customer Churn Prediction — End-to-End ML Project

Predicts whether a bank customer will churn, using the `Customer-Churn-Records.csv` dataset.
Four classification models are trained and compared; the best one (by F1-score) is deployed
in a Streamlit web app.

## Project Structure
```
churn_prediction_project/
│
├── data/
│   └── Customer-Churn-Records.csv
├── notebooks/
│   └── Churn_Prediction_Model_Comparison.ipynb
├── models/
│   ├── churn_best_model.pkl
│   ├── gender_label_encoder.pkl
│   ├── feature_scaler.pkl
│   ├── feature_columns.pkl
│   └── best_model_name.pkl
├── app.py
├── requirements.txt
└── README.md
```

## How it works
1. **Notebook** (`notebooks/Churn_Prediction_Model_Comparison.ipynb`) — loads and explores the
   data, preprocesses it (label encoding, one-hot encoding, scaling), trains Logistic Regression,
   Decision Tree, Random Forest, and Gradient Boosting classifiers, compares them on Accuracy,
   Precision, Recall, F1-score and ROC-AUC, then saves the best model and its preprocessing
   artifacts to `models/`.
2. **App** (`app.py`) — a Streamlit form that collects a customer's details, applies the same
   preprocessing used during training, and returns a churn prediction with probability.

## Running locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Retraining
Open and run `notebooks/Churn_Prediction_Model_Comparison.ipynb` top to bottom — it will
regenerate everything in `models/` from the raw CSV in `data/`.

## Deployment
Deploy directly on [Streamlit Community Cloud](https://streamlit.io/cloud) by pointing it at
this repo and `app.py` as the entry point (free, no extra config needed beyond `requirements.txt`).

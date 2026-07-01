import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        background-color: #0d6efd;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
    }
    .stButton>button:hover { background-color: #0b5ed7; }
    h1 { color: #212529; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    """Load the trained model plus every preprocessing artifact saved by the notebook."""
    model = joblib.load(os.path.join(MODEL_DIR, "churn_best_model.pkl"))
    gender_encoder = joblib.load(os.path.join(MODEL_DIR, "gender_label_encoder.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "feature_scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    model_name = joblib.load(os.path.join(MODEL_DIR, "best_model_name.pkl"))
    return model, gender_encoder, scaler, feature_columns, model_name


def build_input_row(form_values: dict, gender_encoder, feature_columns) -> pd.DataFrame:
    """Turn the raw form inputs into a single-row DataFrame matching the training feature set."""
    gender_encoded = int(gender_encoder.transform([form_values["gender"]])[0])

    row = {
        "CreditScore": form_values["credit_score"],
        "Gender": gender_encoded,
        "Age": form_values["age"],
        "Tenure": form_values["tenure"],
        "Balance": form_values["balance"],
        "NumOfProducts": form_values["num_of_products"],
        "HasCrCard": 1 if form_values["has_crcard"] == "Yes" else 0,
        "IsActiveMember": 1 if form_values["is_active_member"] == "Yes" else 0,
        "EstimatedSalary": form_values["estimated_salary"],
        "Complain": 1 if form_values["complain"] == "Yes" else 0,
        "Satisfaction Score": form_values["satisfaction_score"],
        "Point Earned": form_values["point_earned"],
        "Geography_France": 1 if form_values["geography"] == "France" else 0,
        "Geography_Germany": 1 if form_values["geography"] == "Germany" else 0,
        "Geography_Spain": 1 if form_values["geography"] == "Spain" else 0,
    }
    # Reindex guarantees the exact column order the model was trained on.
    return pd.DataFrame([row])[feature_columns]


st.title("📉 Customer Churn Prediction System")

try:
    model, gender_encoder, scaler, feature_columns, model_name = load_artifacts()
    st.caption(f"Serving model: **{model_name}**")
except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.stop()

st.write("Fill in the customer's details below and click **Predict** to estimate churn risk.")
st.write("---")

with st.form("churn_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Personal Info")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        geography = st.selectbox("Geography (Country)", ["France", "Germany", "Spain"])
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0, step=1000.0)

    with col2:
        st.subheader("🏦 Bank Details")
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        balance = st.number_input("Account Balance", min_value=0.0, value=15000.0, step=500.0)
        num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
        tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=10, value=5)

    with col3:
        st.subheader("⚙️ Activity & Satisfaction")
        has_crcard = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
        complain = st.selectbox("Has Filed a Complaint?", ["No", "Yes"])
        satisfaction_score = st.slider("Satisfaction Score", min_value=1, max_value=5, value=3)
        point_earned = st.number_input("Loyalty Points Earned", min_value=0, value=500, step=50)

    st.markdown("---")
    submitted = st.form_submit_button("🚀 Predict Churn")

if submitted:
    form_values = dict(
        age=age, gender=gender, geography=geography, estimated_salary=estimated_salary,
        credit_score=credit_score, balance=balance, num_of_products=num_of_products,
        tenure=tenure, has_crcard=has_crcard, is_active_member=is_active_member,
        complain=complain, satisfaction_score=satisfaction_score, point_earned=point_earned,
    )

    input_df = build_input_row(form_values, gender_encoder, feature_columns)

    # Logistic Regression was trained on scaled features; tree-based models use raw features.
    if model_name == "Logistic Regression":
        model_input = scaler.transform(input_df)
    else:
        model_input = input_df

    prediction = model.predict(model_input)[0]
    probability = model.predict_proba(model_input)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"🚨 This customer is likely to **CHURN** (probability: {probability:.1%})")
        st.write("Suggested action: proactive retention outreach — a loyalty offer, fee waiver, or personal check-in.")
    else:
        st.success(f"✅ This customer is likely to **STAY** (churn probability: {probability:.1%})")

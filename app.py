import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    model = joblib.load("models/best_model.pkl")
    threshold = joblib.load("models/best_threshold.pkl")
    return model, threshold

model, threshold = load_model()

# Application title
st.title("📊 Telecom Customer Churn Prediction")

st.write(
    "Predict whether a telecom customer is likely to churn using Machine Learning."
)

st.success("Machine Learning model loaded successfully!")

st.divider()

st.subheader("Customer Information")

# Customer inputs
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

with col3:
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly = st.number_input("Monthly Charges", 0.0, 2000.0, 70.0)
    total = st.number_input("Total Charges", 0.0, 100000.0, 840.0)

    st.info("Complete all customer details before predicting.")

# Prediction
if st.button("Predict Customer Churn", type="primary"):

    service_columns = [
        phone, multiple, security, backup,
        protection, support, tv, movies
    ]

    service_count = sum(value == "Yes" for value in service_columns)

    if tenure <= 12:
        tenure_group = "New"
    elif tenure <= 24:
        tenure_group = "Medium"
    elif tenure <= 48:
        tenure_group = "Long"
    else:
        tenure_group = "Very Long"

    total_charges_per_tenure = total / tenure if tenure > 0 else np.nan

    new_customer = int(tenure <= 6)

    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "ServiceCount": service_count,
        "TenureGroup": tenure_group,
        "TotalChargesPerTenure": total_charges_per_tenure,
        "NewCustomer": new_customer
    }])

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if probability >= threshold:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")

    st.metric("Churn Probability", f"{probability * 100:.2f}%")
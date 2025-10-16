import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load("churn_model.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.markdown("Enter customer details to predict churn probability.")

# Sidebar for inputs
st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
partner = st.sidebar.selectbox("Partner", ("Yes", "No"))
dependents = st.sidebar.selectbox("Dependents", ("Yes", "No"))
phone_service = st.sidebar.selectbox("Phone Service", ("Yes", "No"))
multiple_lines = st.sidebar.selectbox("Multiple Lines", ("Yes", "No"))
online_security = st.sidebar.selectbox("Online Security", ("Yes", "No"))
online_backup = st.sidebar.selectbox("Online Backup", ("Yes", "No"))
device_protection = st.sidebar.selectbox("Device Protection Plan", ("Yes", "No"))
tech_support = st.sidebar.selectbox("Tech Support", ("Yes", "No"))
streaming_tv = st.sidebar.selectbox("Streaming TV", ("Yes", "No"))
paperless_billing = st.sidebar.selectbox("Paperless Billing", ("Yes", "No"))

internet_service = st.sidebar.selectbox("Internet Service", ("DSL", "Fiber optic", "None"))
contract = st.sidebar.selectbox("Contract", ("Month-to-month", "One year", "Two year"))
payment_method = st.sidebar.selectbox("Payment Method", (
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
))

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charge = st.sidebar.number_input("Monthly Charge", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)

# Prepare input dataframe (must match training)
def preprocess_input():
    data = {
        "Gender": [1 if gender=="Male" else 0],
        "Partner": [1 if partner=="Yes" else 0],
        "Dependents": [1 if dependents=="Yes" else 0],
        "Phone_Service": [1 if phone_service=="Yes" else 0],
        "Multiple_Lines": [1 if multiple_lines=="Yes" else 0],
        "Online_Security": [1 if online_security=="Yes" else 0],
        "Online_Backup": [1 if online_backup=="Yes" else 0],
        "Device_Protection_Plan": [1 if device_protection=="Yes" else 0],
        "Tech_Support": [1 if tech_support=="Yes" else 0],
        "Streaming_TV": [1 if streaming_tv=="Yes" else 0],
        "Paperless_Billing": [1 if paperless_billing=="Yes" else 0],
        "Tenure_in_Months": [tenure/72],  # scaled
        "Monthly_Charge": [monthly_charge/200],  # scaled
        "Total_Charges": [total_charges/10000],  # scaled
        # One-hot encoding for categorical vars
        "Internet_Service_Fiber optic": [1 if internet_service=="Fiber optic" else 0],
        "Internet_Service_None": [1 if internet_service=="None" else 0],
        "Contract_One year": [1 if contract=="One year" else 0],
        "Contract_Two year": [1 if contract=="Two year" else 0],
        "Payment_Method_Credit card (automatic)": [1 if payment_method=="Credit card (automatic)" else 0],
        "Payment_Method_Electronic check": [1 if payment_method=="Electronic check" else 0],
        "Payment_Method_Mailed check": [1 if payment_method=="Mailed check" else 0]
    }
    return pd.DataFrame(data)

# Predict button
if st.sidebar.button("Predict Churn"):
    input_df = preprocess_input()
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("📌 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Customer is **likely to Churn** (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer will **Stay** (Probability of Churn: {prob:.2f})")

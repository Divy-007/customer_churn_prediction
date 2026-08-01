"""
Streamlit frontend — thin client for the Churn Prediction API.
Run api.py first (uvicorn app.api:app --reload --port 8000), then this.
"""

import streamlit as st
import requests
st.write("TEST — if you see this, the file is loading")

API_URL = "http://localhost:8000/predict"  # update to your deployed URL later

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Predictor")
st.caption("Predicts churn risk and explains why — powered by SHAP + LLM.")

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", 0, 100, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with col2:
        device_protect = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges", 0.0, 150.0, 70.0)
        total_charges = st.number_input("Total Charges", 0.0, 9000.0, 840.0)

    submitted = st.form_submit_button("Predict Churn", use_container_width=True)

if submitted:
    payload = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multiple_lines, "InternetService": internet,
        "OnlineSecurity": online_sec, "OnlineBackup": online_backup,
        "DeviceProtection": device_protect, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    with st.spinner("Predicting and analyzing why..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            churn = result["prediction"] == 1
            prob = result["churn_probability"]

            st.divider()

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Churn Probability", f"{prob:.1%}")
            with col_b:
                st.metric("Prediction", "⚠️ Will Churn" if churn else "✅ Will Stay")

            st.subheader("Why this prediction?")
            st.write(result["explanation"])

            st.subheader("💡 Recommended Action")
            st.info(result["solution"])

            st.subheader("Top Contributing Factors (SHAP)")
            st.bar_chart(result["top_factors"])

        except requests.exceptions.ConnectionError:
            st.error("Couldn't reach the API. Is it running? (`uvicorn app.api:app --reload --port 8000`)")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
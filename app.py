import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="💳 Fraud Detection", layout="wide")

# Load trained model
model = joblib.load("credit_card_fraud_model.pkl")

st.title("💳 Credit Card Fraud Detection App")

st.header("🔍 Manually Enter Transaction Features")

with st.form("manual_form"):
    time = st.number_input("Time", value=0.0)
    v_features = [st.number_input(f"V{i}", value=0.0) for i in range(1, 29)]
    amount = st.number_input("Amount", value=0.0)
    features = [time] + v_features + [amount]

    submit = st.form_submit_button("Predict")

if submit:
    input_array = np.array([features])
    prediction = model.predict(input_array)[0]
    st.success(f"Prediction: {'🚨 Fraud' if prediction == 1 else '✅ Legit'}")

st.header("📂 Or Upload a CSV File")

uploaded_file = st.file_uploader("Upload a CSV with columns: Time, V1–V28, Amount", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("🔍 Preview of Uploaded Data:")
        st.dataframe(df.head())

        predictions = model.predict(df)
        df["Prediction"] = predictions
        df["Result"] = df["Prediction"].map({0: "✅ Legit", 1: "🚨 Fraud"})

        st.success("Predictions completed!")
        st.dataframe(df[["Prediction", "Result"]])

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error: {e}")

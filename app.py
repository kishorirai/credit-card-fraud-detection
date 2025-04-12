import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="💳 Fraud Detection", layout="wide")

# Custom CSS for background and styling
st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
        }
        .stButton>button {
            background-color: #6c63ff;
            color: white;
            border-radius: 8px;
        }
        .main {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <h2 style='text-align: center; color: #6c63ff;'>Madhav Institute of Technology & Science (MITS), Gwalior</h2>
    <h3 style='text-align: center;'>Minor Project: Credit Card Fraud Detection Using Machine Learning</h3>
    <p style='text-align: center;'>By: <b>Kishori Kumari</b></p>
    <hr style="border: 1px solid #6c63ff;" />
""", unsafe_allow_html=True)

# Load the model
model = joblib.load("credit_card_fraud_model.pkl")

st.title("💳 Credit Card Fraud Detection App")

# Tabs for two modes
tab1, tab2 = st.tabs(["📝 Manual Entry", "📁 Upload CSV"])

with tab1:
    st.subheader("🔍 Manually Enter Transaction Features")
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

with tab2:
    st.subheader("📂 Upload a CSV File")
    uploaded_file = st.file_uploader("Upload a CSV with columns: Time, V1–V28, Amount", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Remove 'Class' column if present
            if 'Class' in df.columns:
                df = df.drop(columns=['Class'])

            st.write("🔍 Preview of Uploaded Data:")
            st.dataframe(df.head())

            predictions = model.predict(df)
            df["Prediction"] = predictions
            df["Result"] = df["Prediction"].map({0: "✅ Legit", 1: "🚨 Fraud"})

            st.success("✅ Predictions completed!")
            st.dataframe(df[["Prediction", "Result"]])

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# Footer
st.markdown("""
    <hr />
    <p style='text-align: center; font-size: 0.8rem;'>© 2025 Kishori Kumari | MITS Gwalior</p>
""", unsafe_allow_html=True)

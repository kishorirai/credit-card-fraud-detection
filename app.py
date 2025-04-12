import streamlit as st
import numpy as np
import pandas as pd
import joblib

# CONFIG
st.set_page_config(page_title="💳 Credit Card Fraud Detection", layout="wide")

# LOAD MODEL
model = joblib.load("credit_card_fraud_model.pkl")

# CUSTOM CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    .main {
        background-color: #f6f9fc;
        padding: 30px;
        border-radius: 10px;
    }

    .title {
        background: linear-gradient(to right, #6a11cb, #2575fc);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }

    .stButton>button {
        background-color: #2575fc;
        color: white;
        padding: 10px 16px;
        border-radius: 8px;
        border: none;
    }

    .footer {
        text-align: center;
        padding-top: 20px;
        font-size: 0.85rem;
        color: #888;
    }

    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# TITLE
st.markdown("""
<div class='title'>
    <h2>💳 Credit Card Fraud Detection</h2>
    <p><b>Kishori Kumari</b> | Madhav Institute of Technology and Science (MITS), Gwalior</p>
</div>
""", unsafe_allow_html=True)

# TABS
tab1, tab2 = st.tabs(["📝 Manual Input", "📁 CSV Upload"])

# ---------- TAB 1 ----------
with tab1:
    st.markdown("### 🔍 Manually Enter Transaction Features")
    with st.form("manual_form"):
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            time = st.number_input("🕒 Time", value=0.0)
            v_features = [st.number_input(f"🔢 V{i}", value=0.0) for i in range(1, 29)]
            amount = st.number_input("💰 Amount", value=0.0)
            features = [time] + v_features + [amount]
            st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("🔎 Predict")

    if submitted:
        input_array = np.array([features])
        prediction = model.predict(input_array)[0]
        result = "🚨 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"
        st.markdown(f"<div class='card'><h4>🧾 Result: {result}</h4></div>", unsafe_allow_html=True)

# ---------- TAB 2 ----------
with tab2:
    st.markdown("### 📂 Upload a CSV File")

    uploaded_file = st.file_uploader("Upload CSV with columns: Time, V1–V28, Amount", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if "Class" in df.columns:
                df = df.drop(columns=["Class"])

            st.markdown("#### 👀 Preview of Uploaded Data")
            st.dataframe(df.head())

            predictions = model.predict(df)
            df["Prediction"] = predictions
            df["Result"] = df["Prediction"].map({0: "✅ Legit", 1: "🚨 Fraud"})

            st.success("🎯 Predictions done!")
            st.dataframe(df[["Prediction", "Result"]])

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results", csv, "fraud_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ---------- FOOTER ----------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div class='footer'>
    © 2025 Kishori Kumari | MITS Gwalior · Streamlit App for Minor Project 💼
</div>
""", unsafe_allow_html=True)


import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# PAGE CONFIG
st.set_page_config(page_title="💳 Credit Card Fraud Detection", layout="wide")

# LOAD MODEL
model = joblib.load("credit_card_fraud_model.pkl")


bg_color = "#f6f9fc"
card_color = "white"
text_color = "#003366"
header_bg = "linear-gradient(to right, #b3cde0, #f1f1f1)"

st.markdown(f"""
<style>
    html, body, [class*="css"]  {{
        font-family: 'Poppins', sans-serif;
        background-color: {bg_color};
        color: {text_color};
    }}

    .main {{
        background-color: {bg_color};
        padding: 30px;
        border-radius: 10px;
    }}

    .title {{
        background: {header_bg};
        padding: 2rem;
        border-radius: 15px;
        color: {text_color};
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
    }}

    .stButton>button {{
        background-color: #7baedc;
        color: white;
        padding: 10px 16px;
        border-radius: 8px;
        border: none;
    }}

    .footer {{
        text-align: center;
        padding-top: 20px;
        font-size: 0.85rem;
        color: #888;
    }}

    .card {{
        background-color: {card_color};
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
        color: {text_color};
    }}
</style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.markdown(f"""
<div class='title'>
    <h2>💳 Credit Card Fraud Detection</h2>
</div>
""", unsafe_allow_html=True)

# ---- TABS ----
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Manual Input", "📁 CSV Upload", ])



   
    # ---------- TAB 1: Manual Input ----------
with tab1:
    st.markdown("### 🔍 Manually Enter Transaction Features")
    with st.form("manual_form"):
        st.markdown("#### ⏱️ Enter Time")
        time = st.number_input("Time", value=0.0)

        st.markdown("#### 🧮 Enter Feature Values (V1 - V28)")
        col1, col2 = st.columns(2)

        with col1:
            v1 = st.number_input("V1", value=0.0)
            v2 = st.number_input("V2", value=0.0)
            v3 = st.number_input("V3", value=0.0)
            v4 = st.number_input("V4", value=0.0)
            v5 = st.number_input("V5", value=0.0)
            v6 = st.number_input("V6", value=0.0)
            v7 = st.number_input("V7", value=0.0)
            v8 = st.number_input("V8", value=0.0)
            v9 = st.number_input("V9", value=0.0)
            v10 = st.number_input("V10", value=0.0)
            v11 = st.number_input("V11", value=0.0)
            v12 = st.number_input("V12", value=0.0)
            v13 = st.number_input("V13", value=0.0)
            v14 = st.number_input("V14", value=0.0)

        with col2:
            v15 = st.number_input("V15", value=0.0)
            v16 = st.number_input("V16", value=0.0)
            v17 = st.number_input("V17", value=0.0)
            v18 = st.number_input("V18", value=0.0)
            v19 = st.number_input("V19", value=0.0)
            v20 = st.number_input("V20", value=0.0)
            v21 = st.number_input("V21", value=0.0)
            v22 = st.number_input("V22", value=0.0)
            v23 = st.number_input("V23", value=0.0)
            v24 = st.number_input("V24", value=0.0)
            v25 = st.number_input("V25", value=0.0)
            v26 = st.number_input("V26", value=0.0)
            v27 = st.number_input("V27", value=0.0)
            v28 = st.number_input("V28", value=0.0)

        st.markdown("#### 💰 Enter Transaction Amount")
        amount = st.number_input("Amount", value=0.0)

        submitted = st.form_submit_button("🔎 Predict")

    if submitted:
        features = [time] + [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                             v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                             v21, v22, v23, v24, v25, v26, v27, v28, amount]

        input_array = np.array([features])
        prediction = model.predict(input_array)[0]
        prediction_prob = model.predict_proba(input_array)[0][1]
        result = "🚨 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"
        confidence = f"{prediction_prob * 100:.2f}%"
        fraud_risk_score = int(prediction_prob * 100)

        st.markdown(f"### 🧾 Result: {result}")
        st.markdown(f"**Confidence Level:** {confidence}")
        st.markdown(f"**Fraud Risk Score:** {fraud_risk_score}")



      
# ---------- TAB 2: CSV Upload ---------- 
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
            prediction_probs = model.predict_proba(df)[:, 1]
            df["Prediction"] = predictions
            df["Confidence"] = prediction_probs * 100
            df["Result"] = df["Prediction"].map({0: "✅ Legit", 1: "🚨 Fraud"})

            st.success("🎯 Predictions done!")
            st.dataframe(df[["Prediction", "Confidence", "Result"]])

            # Show Cross-validation scores
            cv_scores = cross_val_score(model, df.drop(columns=["Prediction", "Result"]), df["Prediction"], cv=5)
            st.markdown(f"**Model Cross-validation Score**: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results", csv, "fraud_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")


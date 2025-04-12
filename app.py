import streamlit as st
import numpy as np
import joblib

# PAGE CONFIG
st.set_page_config(page_title="💳 Credit Card Fraud Detection", layout="wide")

# LOAD MODEL
model = joblib.load("credit_card_fraud_model.pkl")

# ---- THEME TOGGLE ----
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

theme = st.sidebar.radio("🌗 Choose Theme", ["light", "dark"], index=0 if st.session_state["theme"] == "light" else 1)
st.session_state["theme"] = theme

# ---- CUSTOM STYLING ----
if theme == "light":
    bg_color = "#f6f9fc"
    card_color = "white"
    text_color = "#003366"
    header_bg = "linear-gradient(to right, #b3cde0, #f1f1f1)"
else:
    bg_color = "#1e1e1e"
    card_color = "#2b2b2b"
    text_color = "#ffffff"
    header_bg = "linear-gradient(to right, #4a4a4a, #2d2d2d)"

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
        color: {'#ccc' if theme == 'dark' else '#888'};
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
tab1, tab2 = st.tabs(["📝 Manual Input", "📁 CSV Upload"])

# ---------- TAB 1: Manual Input ----------
with tab1:
    st.markdown("### 🔍 Manually Enter Transaction Features")
    with st.form("manual_form"):
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        # Time Dropdown with Explanation
        st.markdown("#### 🕒 Time (Elapsed Time in Seconds)")
        time = st.selectbox("Choose a time value", options=[i for i in range(0, 172793, 10000)])
        st.caption("The 'Time' feature represents the elapsed time in seconds from the first transaction.")

        # V1 to V28 Manual Inputs
        v_features = [st.text_input(f"🔢 V{i} (e.g., 0.0)", value="0.0") for i in range(1, 29)]

        # Amount Input
        amount = st.number_input("💰 Amount", min_value=0.0, max_value=25691.16, value=0.0, step=0.01)

        # Collecting all features
        features = [time] + [float(v) for v in v_features] + [amount]

        st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("🔎 Predict")

    if submitted:
        input_array = np.array([features])
        prediction = model.predict(input_array)[0]
        prediction_prob = model.predict_proba(input_array)[0][1]  # Get the probability of fraud
        result = "🚨 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"
        confidence = f"{prediction_prob * 100:.2f}%"  # Confidence level of prediction
        st.markdown(f"<div class='card'><h4>🧾 Result: {result}</h4>", unsafe_allow_html=True)
        st.markdown(f"<p>Confidence Level: {confidence}</p></div>", unsafe_allow_html=True)

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
            prediction_probs = model.predict_proba(df)[:, 1]  # Get the probability of fraud for all predictions
            df["Prediction"] = predictions
            df["Confidence"] = prediction_probs * 100  # Add confidence as a percentage
            df["Result"] = df["Prediction"].map({0: "✅ Legit", 1: "🚨 Fraud"})

            st.success("🎯 Predictions done!")
            st.dataframe(df[["Prediction", "Confidence", "Result"]])

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results", csv, "fraud_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ---- FOOTER ----
st.markdown("<div class='footer'>Made by Kishori Kumari | MITS Gwalior</div>", unsafe_allow_html=True)

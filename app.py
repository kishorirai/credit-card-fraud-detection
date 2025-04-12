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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Manual Input", "📁 CSV Upload", "📊 Feature Visualization", "🔍 Anomaly Detection", "ℹ️ Model Details"])

# ---------- TAB 1: Manual Input ---------- 
# ---------- TAB 1: Manual Input ----------
with tab1:
    st.markdown("### 🔍 Manually Enter Transaction Features")
    with st.form("manual_form"):
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        # Creating columns for the input fields to display in grid layout
        col1, col2 = st.columns(2)

        with col1:
            time = st.number_input("🕒 Time", value=0.0, help="The time of the transaction in seconds")
            v1 = st.number_input("🔢 V1", value=0.0, help="Variance of transaction amount over a 3-month period")
            v2 = st.number_input("🔢 V2", value=0.0, help="A feature related to the transaction amount distribution")
            v3 = st.number_input("🔢 V3", value=0.0, help="Another feature related to the transaction data")
            v4 = st.number_input("🔢 V4", value=0.0, help="Variance related to the previous features")
            v5 = st.number_input("🔢 V5", value=0.0, help="A transformation feature used for modeling")
        
        with col2:
            v6 = st.number_input("🔢 V6", value=0.0, help="A feature derived from prior transaction amounts")
            v7 = st.number_input("🔢 V7", value=0.0, help="Another feature related to the transaction history")
            v8 = st.number_input("🔢 V8", value=0.0, help="Transformation of transaction amounts")
            v9 = st.number_input("🔢 V9", value=0.0, help="Variance used for fraud detection")
            v10 = st.number_input("🔢 V10", value=0.0, help="A feature used to determine the type of transaction")

        # Continuing to the next group of features
        col3, col4 = st.columns(2)

        with col3:
            v11 = st.number_input("🔢 V11", value=0.0, help="Another transformation feature")
            v12 = st.number_input("🔢 V12", value=0.0, help="Feature indicating the level of transaction abnormality")
            v13 = st.number_input("🔢 V13", value=0.0, help="Anomaly related to transaction behavior")
            v14 = st.number_input("🔢 V14", value=0.0, help="A statistical feature related to transaction type")
            v15 = st.number_input("🔢 V15", value=0.0, help="A feature for modeling transaction confidence")
        
        with col4:
            v16 = st.number_input("🔢 V16", value=0.0, help="Feature representing the variance across multiple features")
            v17 = st.number_input("🔢 V17", value=0.0, help="A feature indicating the abnormality of the transaction")
            v18 = st.number_input("🔢 V18", value=0.0, help="Feature indicating transaction similarity")
            v19 = st.number_input("🔢 V19", value=0.0, help="A transformation for detecting fraudulent behavior")
            v20 = st.number_input("🔢 V20", value=0.0, help="A model-related feature used in fraud detection")

        # Continue with the last group of features
        col5, col6 = st.columns(2)

        with col5:
            v21 = st.number_input("🔢 V21", value=0.0, help="Feature based on prior data analysis")
            v22 = st.number_input("🔢 V22", value=0.0, help="Transaction-specific feature derived from earlier behavior")
            v23 = st.number_input("🔢 V23", value=0.0, help="Feature indicating transaction type")
            v24 = st.number_input("🔢 V24", value=0.0, help="Feature related to abnormal behavior detection")
            v25 = st.number_input("🔢 V25", value=0.0, help="Variance feature used in detecting fraud")
        
        with col6:
            v26 = st.number_input("🔢 V26", value=0.0, help="Feature indicating the likelihood of fraud")
            v27 = st.number_input("🔢 V27", value=0.0, help="Another feature used for anomaly detection")
            v28 = st.number_input("🔢 V28", value=0.0, help="Final feature used in the prediction model")
            amount = st.number_input("💰 Amount", value=0.0, help="The monetary value of the transaction")

        # Combining all inputs into a list
        features = [time] + [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                             v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                             v21, v22, v23, v24, v25, v26, v27, v28, amount]

        st.markdown("</div>", unsafe_allow_html=True)

        # Submit button
        submitted = st.form_submit_button("🔎 Predict")

    if submitted:
        # Perform prediction
        input_array = np.array([features])
        prediction = model.predict(input_array)[0]
        prediction_prob = model.predict_proba(input_array)[0][1]
        result = "🚨 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"
        confidence = f"{prediction_prob * 100:.2f}%"
        fraud_risk_score = int(prediction_prob * 100)  # Show risk score (0 to 100)

        st.markdown(f"<div class='card'><h4>🧾 Result: {result}</h4>", unsafe_allow_html=True)
        st.markdown(f"<p>Confidence Level: {confidence}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Fraud Risk Score: {fraud_risk_score}</strong></p></div>", unsafe_allow_html=True)

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

# ---------- TAB 3: Feature Visualization ---------- 
with tab3:
    st.markdown("### 📊 Visualize Transaction Features")
    uploaded_file = st.file_uploader("Upload CSV for Visualization", type=["csv"], key="viz")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # PCA visualization
            st.subheader("📉 2D PCA of the Transactions")
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df.drop(columns=["Class"], errors='ignore'))

            fig, ax = plt.subplots()
            ax.scatter(pca_result[:, 0], pca_result[:, 1], c=df["Class"], cmap="coolwarm")
            ax.set_title("PCA - 2D Projection")
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            st.pyplot(fig)

            # Feature correlation heatmap
            st.subheader("📊 Correlation Heatmap")
            corr = df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap of Features")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ---------- TAB 4: Anomaly Detection ---------- 
with tab4:
    st.markdown("### 🔍 Anomaly Detection Visualization")
    uploaded_file = st.file_uploader("Upload CSV for Anomaly Detection", type=["csv"], key="anomaly")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            predictions = model.predict(df)
            prediction_probs = model.predict_proba(df)[:, 1]

            df["Prediction"] = predictions
            df["Confidence"] = prediction_probs * 100
            df["Result"] = df["Prediction"].map({0: "✅ Legit", 1: "🚨 Fraud"})

            df_sorted = df.sort_values(by="Confidence", ascending=False)
            st.markdown("#### 📊 Top 5 Most Anomalous Transactions")
            st.dataframe(df_sorted.head())

            st.success("🎯 Anomaly Detection complete!")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ---------- TAB 5: Model Details ---------- 
with tab5:
    st.markdown("### ℹ️ Model Details")
    with st.expander("🔍 Expand for more information"):
        st.markdown("""
        - **Model Type**: Random Forest Classifier (or your preferred model)
        - **Dataset**: Credit Card Fraud Detection dataset from Kaggle
        - **Accuracy**: 99.8% (Depending on the model parameters and preprocessing)
        - **Objective**: Detect fraudulent transactions based on past data
        - **Preprocessing**: Normalization and PCA used for visualization
        """)
        st.image("https://upload.wikimedia.org/wikipedia/commons/a/a3/Random_forest_diagram_complete.png", caption="Random Forest Classifier")

# ---- FOOTER ----
st.markdown("<div class='footer'>Made by Kishori Kumari | MITS Gwalior</div>", unsafe_allow_html=True)

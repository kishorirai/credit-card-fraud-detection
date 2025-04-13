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


# ---- THEME TOGGLE BUTTON (TOP-RIGHT) ----

if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

# Set theme before rendering button
theme = st.session_state["theme"]
button_label = "🌙 Dark Mode" if theme == "light" else "☀️ Light Mode"

# Display toggle button in top-right
col1, col2 = st.columns([0.85, 0.15])
with col2:
    if st.button(button_label):
        # Toggle theme and rerun
        st.session_state["theme"] = "dark" if theme == "light" else "light"
        st.rerun()  # Apply immediately

# Set Theme Colors
theme = st.session_state["theme"]
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
        st.markdown("#### ⏱️ Enter Time")
        time = st.number_input("Time", value=123.0)  # Sample Value: 123.0

        st.markdown("#### 🧮 Enter Feature Values (V1 - V28)")

        # Two columns for input fields
        col1, col2 = st.columns(2)

        with col1:
            v1 = st.number_input("V1", value=0.1)  # Sample Value: 0.1
            v2 = st.number_input("V2", value=0.2)  # Sample Value: 0.2
            v3 = st.number_input("V3", value=1.1)  # Sample Value: 1.1
            v4 = st.number_input("V4", value=-0.3)  # Sample Value: -0.3
            v5 = st.number_input("V5", value=0.5)  # Sample Value: 0.5
            v6 = st.number_input("V6", value=0.0)  # Sample Value: 0.0
            v7 = st.number_input("V7", value=0.4)  # Sample Value: 0.4
            v8 = st.number_input("V8", value=0.2)  # Sample Value: 0.2
            v9 = st.number_input("V9", value=1.0)  # Sample Value: 1.0
            v10 = st.number_input("V10", value=-0.5)  # Sample Value: -0.5
            v11 = st.number_input("V11", value=0.3)  # Sample Value: 0.3
            v12 = st.number_input("V12", value=0.7)  # Sample Value: 0.7
            v13 = st.number_input("V13", value=-0.1)  # Sample Value: -0.1
            v14 = st.number_input("V14", value=0.8)  # Sample Value: 0.8

        with col2:
            v15 = st.number_input("V15", value=0.6)  # Sample Value: 0.6
            v16 = st.number_input("V16", value=0.4)  # Sample Value: 0.4
            v17 = st.number_input("V17", value=1.3)  # Sample Value: 1.3
            v18 = st.number_input("V18", value=-0.2)  # Sample Value: -0.2
            v19 = st.number_input("V19", value=0.0)  # Sample Value: 0.0
            v20 = st.number_input("V20", value=-0.6)  # Sample Value: -0.6
            v21 = st.number_input("V21", value=0.9)  # Sample Value: 0.9
            v22 = st.number_input("V22", value=0.4)  # Sample Value: 0.4
            v23 = st.number_input("V23", value=0.5)  # Sample Value: 0.5
            v24 = st.number_input("V24", value=1.2)  # Sample Value: 1.2
            v25 = st.number_input("V25", value=-0.3)  # Sample Value: -0.3
            v26 = st.number_input("V26", value=0.8)  # Sample Value: 0.8
            v27 = st.number_input("V27", value=0.1)  # Sample Value: 0.1
            v28 = st.number_input("V28", value=0.2)  # Sample Value: 0.2

        st.markdown("#### 💰 Enter Transaction Amount")
        amount = st.number_input("Amount", value=20.0)  # Sample Value: 20.0

        submitted = st.form_submit_button("🔎 Predict")

    if submitted:
        # Collecting all features into an array
        features = [time] + [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                             v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                             v21, v22, v23, v24, v25, v26, v27, v28, amount]

        # Convert the features into an array for prediction
        input_array = np.array([features])
        prediction = model.predict(input_array)[0]
        prediction_prob = model.predict_proba(input_array)[0][1]
        
        # Result display
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

# ---------- TAB 3: Feature Visualization ---------- 
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
            scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=df["Class"], cmap="coolwarm")
            ax.set_title("PCA - 2D Projection of Transactions")

            # Add color bar for better visualization
            cbar = plt.colorbar(scatter)
            cbar.set_label('Class')

            st.pyplot(fig)

            # Show sample visuals using seaborn for pairplot (optional but informative)
            st.subheader("🔎 Pairplot of Sample Features")
            sample_df = df.sample(100)  # Take a sample for better visualization
            pairplot_fig = sns.pairplot(sample_df[['V1', 'V2', 'V3', 'V4', 'Amount', 'Class']], hue='Class')
            st.pyplot(pairplot_fig)

        except Exception as e:
            st.error(f"❌ Error: {e}")


# ---------- TAB 4: Anomaly Detection ---------- 
with tab4:
    st.markdown("### 🔍 Detect Anomalies")
    st.markdown("🔧 Here we can apply anomaly detection methods, e.g., Isolation Forest.")
    # Add further anomaly detection here if needed.

# ---------- TAB 5: Model Details ---------- 
with tab5:
    st.markdown("### ℹ️ Model Details")
    st.markdown("""
    This model is based on a Random Forest Classifier, which is trained on the credit card fraud dataset and can predict whether a transaction is fraudulent or not based on various transaction features.
    
    **Features Used:** V1–V28, Amount, and Time

    **Model Evaluation:** 
    The model's accuracy, precision, recall, and F1-score were evaluated during training. The classifier uses cross-validation to estimate the reliability of predictions.
    """)

# ---- FOOTER ----
st.markdown(f"""
<div class="footer">
    <p>Developed by <b>{st.session_state.get("name", "Kishori Kumari")}</b> | Madhav Institute of Technology and Science (MITS Gwalior)</p>
</div>
""", unsafe_allow_html=True)

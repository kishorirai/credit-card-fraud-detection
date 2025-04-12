import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Load the trained model
model = pickle.load(open("credit_card_fraud_model.pkl", "rb"))

# Set the page layout
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Title and description
st.title("Credit Card Fraud Detection App")
st.write("This application detects fraudulent credit card transactions based on a trained machine learning model.")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Manual Input", "CSV Upload", "Feature Visualization", "Anomaly Detection", "Model Details"])

# --- Tab 1: Manual Input ---
with tab1:
    st.markdown("### 📝 Manual Input")
    st.write("Please enter the details of the transaction below:")

    time = st.number_input("Time", min_value=0.0, value=0.0)
    v1 = st.number_input("V1", value=0.0)
    v2 = st.number_input("V2", value=0.0)
    v3 = st.number_input("V3", value=0.0)
    v28 = st.number_input("V28", value=0.0)
    amount = st.number_input("Amount", min_value=0.0, value=0.0)

    input_data = np.array([[time, v1, v2, v3, v28, amount]])

    # Standardize the data
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    if st.button("Predict Fraud"):
        prediction = model.predict(input_data_scaled)
        if prediction == 1:
            st.success("✅ This transaction is **Not Fraud**")
        else:
            st.error("🚨 This transaction is **Fraudulent**")

# --- Tab 2: CSV Upload ---
with tab2:
    st.markdown("### 📤 CSV Upload")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Data preview:")
            st.dataframe(df.head())

            st.write("Predicting fraud for uploaded data...")

            features = df.drop(columns=["Class"], errors="ignore")
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            predictions = model.predict(features_scaled)
            df['Predictions'] = predictions

            df['Prediction_Label'] = df['Predictions'].map({1: 'Not Fraud', 0: 'Fraudulent'})
            st.dataframe(df[['Prediction_Label']])

            # Download results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# --- Tab 3: Feature Visualization ---
with tab3:
    st.markdown("### 📊 Feature Visualization")
    st.write("Visualize the distribution of transaction features.")

    df = pd.read_csv("creditcard.csv")  # Replace with the actual dataset path
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Feature Selection
    selected_features = st.multiselect("Select features to visualize", df.columns, default=df.columns[:3])

    # Visualization
    fig, ax = plt.subplots()
    for feature in selected_features:
        ax.hist(df[feature], bins=30, alpha=0.7, label=feature)
    
    ax.set_title("Feature Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

# --- Tab 4: Anomaly Detection ---
with tab4:
    st.markdown("### 🧪 Unsupervised Anomaly Detection using Isolation Forest")
    anomaly_file = st.file_uploader("Upload CSV for Anomaly Detection", type=["csv"], key="anomaly")

    if anomaly_file is not None:
        try:
            df = pd.read_csv(anomaly_file)
            features_only = df.drop(columns=["Class"], errors="ignore")

            # Standardize the data
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_only)

            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=0.01, random_state=42)
            anomalies = iso_forest.fit_predict(features_scaled)

            df["Anomaly"] = anomalies
            df["Anomaly"] = df["Anomaly"].map({1: "✅ Normal", -1: "🚨 Anomaly"})

            st.success("✅ Anomaly Detection Completed!")
            st.dataframe(df[["Anomaly"]].value_counts().reset_index().rename(columns={0: "Count"}))

            # 2D PCA Plot to visualize anomalies
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(features_scaled)

            fig, ax = plt.subplots()
            ax.scatter(pca_result[:, 0], pca_result[:, 1], c=(anomalies == -1), cmap="coolwarm", s=15)
            ax.set_title("Anomaly Detection (Isolation Forest)")
            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            st.pyplot(fig)

            # Download results
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Anomaly Results", csv, "anomaly_detection.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# --- Tab 5: Model Details ---
with tab5:
    st.markdown("### ℹ️ Model Information and Metrics")
    st.write("This app uses a trained model to classify credit card transactions as fraudulent or legitimate based on anonymized features (V1-V28), time, and amount.")

    st.markdown("#### 📌 Model Details")
    st.markdown("""
    - **Model Type:** Logistic Regression / XGBoost / (your model name)
    - **Input Features:** Time, V1–V28, Amount
    - **Output:** Binary classification (Fraud / Not Fraud)
    - **Training Accuracy:** ~99.9%
    - **Dataset:** [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
    - **Toolkits:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
    """)

    st.markdown("#### 📉 Feature Importance (if available)")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        imp_df = imp_df.sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=imp_df.head(10), x="Importance", y="Feature", palette="Blues_r", ax=ax)
        ax.set_title("Top 10 Feature Importances")
        st.pyplot(fig)
    else:
        st.info("ℹ️ This model does not support feature importance extraction.")

    st.markdown("---")
    st.markdown(f"<div class='footer'>Made with ❤️ by **Kishori Kumari** | MITS Gwalior</div>", unsafe_allow_html=True)

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import os
from datetime import datetime
import altair as alt

# Set page config
st.set_page_config(
    page_title="💳 Credit Card Fraud Detection", 
    layout="wide", 
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/yourusername/credit-fraud-detection',
        'Report a bug': 'https://github.com/yourusername/credit-fraud-detection/issues',
        'About': 'Credit Card Fraud Detection App v1.0'
    }
)

# Directory to save uploaded files
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load("credit_card_fraud_model.pkl")

model = load_model()

# -------- THEME TOGGLE ------------
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

with st.sidebar:
    st.sidebar.title("⚙️ Settings")
    theme = st.sidebar.radio("🌗 Theme", ["Light", "Dark"], index=0 if st.session_state["theme"] == "light" else 1,
                            key="theme_toggle")
    st.session_state["theme"] = theme.lower()
    
    st.sidebar.divider()
    st.sidebar.markdown("### 📊 Statistics")
    if "total_analyzed" not in st.session_state:
        st.session_state["total_analyzed"] = 0
    if "fraud_detected" not in st.session_state:
        st.session_state["fraud_detected"] = 0
        
    st.sidebar.metric("Total Transactions Analyzed", f"{st.session_state['total_analyzed']:,}")
    st.sidebar.metric("Fraud Detected", f"{st.session_state['fraud_detected']:,}")
    
    st.sidebar.divider()
    st.sidebar.info("This app uses machine learning to detect fraudulent credit card transactions.")

# ---- CUSTOM STYLING --------------
theme = st.session_state["theme"]
if theme == "light":
    primary_color = "#3366cc"
    secondary_color = "#7baedc"
    bg_color = "#f9fafb"
    card_color = "white"
    text_color = "#333333"
    header_bg = "linear-gradient(to right, #6190E8, #A7BFE8)"
    accent_color = "#ff6b6b"
else:
    primary_color = "#4d79ff"
    secondary_color = "#7289da"
    bg_color = "#1a1a1a"
    card_color = "#2d2d2d"
    text_color = "#ffffff"
    header_bg = "linear-gradient(to right, #3d4e81, #5753C9)"
    accent_color = "#ff6b6b"

st.markdown(f"""
<style>
    html, body, [class*="css"]  {{
        font-family: 'Poppins', sans-serif;
        background-color: {bg_color};
        color: {text_color};
    }}

    section[data-testid="stSidebar"] {{
        background-color: {'#f0f2f6' if theme == 'light' else '#2b2b2b'};
        border-right: 1px solid {'#e6e6e6' if theme == 'light' else '#3d3d3d'};
    }}

    .main {{
        background-color: {bg_color};
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: {primary_color};
        font-weight: 600;
    }}

    .title-container {{
        background: {header_bg};
        padding: 2rem 2.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }}

    .title-container h1 {{
        color: white;
        margin-bottom: 0.5rem;
        font-size: 2.2rem;
        font-weight: 700;
    }}
    
    .title-container p {{
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        max-width: 800px;
        line-height: 1.5;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: {'#f1f3f9' if theme == 'light' else '#333'};
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        font-weight: 500;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {primary_color} !important;
        color: white !important;
    }}

    [data-testid="stFileUploader"] {{
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
        border: 1px dashed {'#ccc' if theme == 'light' else '#555'};
        background-color: {'#fcfcfc' if theme == 'light' else '#252525'};
    }}
    
    .stat-card {{
        background-color: {card_color};
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
        transition: transform 0.2s;
        border: 1px solid {'#eaeaea' if theme == 'light' else '#3a3a3a'};
    }}
    
    .stat-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0px 6px 12px rgba(0,0,0,0.1);
    }}
    
    .stat-card h3 {{
        margin-top: 0;
        font-size: 1.2rem;
        font-weight: 600;
    }}
    
    .stat-card .value {{
        font-size: 2rem;
        font-weight: 700;
        color: {primary_color};
    }}
    
    .fraud {{
        color: {accent_color} !important;
    }}
    
    .legit {{
        color: #0cc177 !important;
    }}
    
    .alert {{
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }}
    
    .alert-danger {{
        background-color: {'#ffe5e5' if theme == 'light' else '#4a1414'};
        border-left: 4px solid {accent_color};
        color: {'#d32f2f' if theme == 'light' else '#ff8a80'};
    }}
    
    .alert-success {{
        background-color: {'#e5ffe5' if theme == 'light' else '#143d14'};
        border-left: 4px solid #0cc177;
        color: {'#2e7d32' if theme == 'light' else '#81c784'};
    }}
    
    .alert-info {{
        background-color: {'#e5f6ff' if theme == 'light' else '#143a4c'};
        border-left: 4px solid {primary_color};
        color: {'#0288d1' if theme == 'light' else '#4fc3f7'};
    }}
    
    .stDownloadButton button {{
        background-color: {primary_color};
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        border: none;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
        transition: all 0.2s;
    }}
    
    .stDownloadButton button:hover {{
        background-color: {secondary_color};
        box-shadow: 0px 4px 8px rgba(0,0,0,0.15);
    }}
    
    div[data-testid="stDataFrame"] {{
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid {'#eaeaea' if theme == 'light' else '#3a3a3a'};
    }}
    
    .footer {{
        text-align: center;
        margin-top: 4rem;
        padding: 2rem 0;
        font-size: 0.9rem;
        border-top: 1px solid {'#eaeaea' if theme == 'light' else '#3a3a3a'};
        color: {'#888' if theme == 'light' else '#aaa'};
    }}
</style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.markdown(f"""
<div class='title-container'>
    <h1>💳 Credit Card Fraud Detection</h1>
    <p>An advanced machine learning system to detect fraudulent credit card transactions with high accuracy.</p>
</div>
""", unsafe_allow_html=True)

# ---- TABS ----
tab1, tab2, tab3, tab4 = st.tabs([
    "📁 CSV Upload", 
    "📊 Feature Visualization", 
    "🔍 Anomaly Detection", 
    "ℹ️ Model Details"
])

# ---------------------- Tab 1: CSV Upload ---------------------
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📂 Upload Your Transaction Data")
        st.markdown("""
        <div class='alert alert-info'>
        Upload your CSV file with transaction data to detect potential fraud activities. 
        The file should contain the standard features (Time, V1-V28, Amount).
        </div>
        """, unsafe_allow_html=True)
        
        # Upload file
        uploaded_file = st.file_uploader(
            "Drag and drop your CSV file here", 
            type=["csv"], 
            key="fraud_csv",
            help="Make sure your CSV contains the required columns: Time, V1-V28, Amount"
        )

    with col2:
        st.markdown("### 🔢 Sample Format")
        st.markdown("""
        Your CSV should include:
        - `Time`: Seconds elapsed between transactions
        - `V1-V28`: PCA transformed features
        - `Amount`: Transaction amount
        - `Class` (optional): Known fraud labels
        """)
        
        # Sample data preview
        sample_data = {
            'Time': [0, 0, 1, 2, 3],
            'V1': [-1.3598, -1.3598, -1.3598, -1.3598, -1.3598],
            'V2': [-0.0728, -0.0728, -0.0728, -0.0728, -0.0728],
            'Amount': [149.62, 2.69, 378.66, 123.50, 69.99]
        }
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)

    if uploaded_file is not None:
        try:
            with st.spinner("🔄 Processing your file..."):
                df = pd.read_csv(uploaded_file)

                # Check for required columns
                required_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.markdown(f"""
                    <div class='alert alert-danger'>
                    <strong>❌ Missing required columns:</strong> {', '.join(missing_cols)}
                    </div>
                    """, unsafe_allow_html=True)
                    st.stop()

                # Run model predictions
                X = df[required_cols]
                predictions = model.predict(X)
                prediction_probs = model.predict_proba(X)[:, 1]

                # Add results to dataframe
                df['Prediction'] = predictions
                df['Confidence'] = prediction_probs * 100
                df['Result'] = df['Prediction'].map({0: '✅ Legit', 1: '🚨 Fraud'})

                # Update session state statistics
                st.session_state["total_analyzed"] += len(df)
                st.session_state["fraud_detected"] += df['Prediction'].sum()

                # Save processed result
                filename = f"fraud_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                file_path = os.path.join(UPLOAD_DIR, filename)
                df.to_csv(file_path, index=False)
                st.session_state["last_uploaded"] = file_path

                # Success message
                st.markdown(f"""
                <div class='alert alert-success'>
                <strong>✅ Analysis Complete!</strong> {len(df):,} transactions analyzed with {int(df['Prediction'].sum()):,} potential fraud(s) detected.
                </div>
                """, unsafe_allow_html=True)

                # Results tabs
                result_tab1, result_tab2, result_tab3 = st.tabs(["Results Overview", "Data Preview", "Charts"])
                
                with result_tab1:
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.markdown(f"""
                        <div class='stat-card'>
                            <h3>Total Transactions</h3>
                            <div class='value'>{len(df):,}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with metric_col2:
                        fraud_count = int(df['Prediction'].sum())
                        st.markdown(f"""
                        <div class='stat-card'>
                            <h3>Fraud Detected</h3>
                            <div class='value fraud'>{fraud_count:,}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with metric_col3:
                        legit_count = len(df) - fraud_count
                        fraud_rate = fraud_count / len(df) * 100
                        st.markdown(f"""
                        <div class='stat-card'>
                            <h3>Fraud Rate</h3>
                            <div class='value'>{fraud_rate:.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Download button
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Complete Results", 
                        csv_data, 
                        filename, 
                        "text/csv",
                        help="Download full analysis with fraud predictions"
                    )
                
                with result_tab2:
                    st.subheader("Data Preview")
                    
                    # Create color-coded dataframe
                    df_display = df[['Time', 'Amount', 'Prediction', 'Confidence', 'Result']].head(10)
                    
                    # Set display options for better formatting
                    pd.set_option('display.float_format', '{:.2f}'.format)
                    
                    # Display the dataframe
                    st.dataframe(
                        df_display,
                        use_container_width=True,
                        column_config={
                            "Confidence": st.column_config.ProgressColumn(
                                "Confidence",
                                help="Probability of fraud",
                                format="%.2f%%",
                                min_value=0,
                                max_value=100,
                            ),
                            "Result": st.column_config.TextColumn(
                                "Result",
                                help="Fraud detection result"
                            )
                        }
                    )
                    
                    if fraud_count > 0:
                        st.subheader("Fraudulent Transactions")
                        st.dataframe(
                            df[df['Prediction'] == 1][['Time', 'Amount', 'Confidence', 'Result']].sort_values('Confidence', ascending=False),
                            use_container_width=True
                        )
                
                with result_tab3:
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        # Donut chart for fraud vs legitimate transactions
                        fig, ax = plt.subplots(figsize=(5, 5))
                        colors = ['#ff6b6b', '#63cdda'] if theme == 'light' else ['#ff6b6b', '#3d7ea6']
                        
                        # Create donut chart
                        ax.pie(
                            [fraud_count, legit_count],
                            labels=['Fraud', 'Legitimate'],
                            autopct='%1.1f%%',
                            startangle=90,
                            colors=colors,
                            wedgeprops=dict(width=0.5)
                        )
                        
                        centre_circle = plt.Circle((0, 0), 0.3, fc='white' if theme == 'light' else '#1a1a1a')
                        fig.gca().add_artist(centre_circle)
                        
                        ax.axis('equal')
                        ax.set_title("Transaction Distribution", fontsize=14, pad=20)
                        st.pyplot(fig)
                    
                    with chart_col2:
                        # Confidence distribution histogram
                        fig2, ax2 = plt.subplots(figsize=(5, 5))
                        sns.histplot(
                            data=df, 
                            x='Confidence', 
                            hue='Prediction',
                            bins=20,
                            palette=['#63cdda', '#ff6b6b'] if theme == 'light' else ['#3d7ea6', '#ff6b6b'],
                            ax=ax2
                        )
                        ax2.set_title("Fraud Confidence Distribution", fontsize=14)
                        ax2.set_xlabel("Confidence Score (%)")
                        ax2.set_ylabel("Count")
                        st.pyplot(fig2)
                    
                    # Amount vs. Confidence scatterplot
                    st.subheader("Transaction Amount vs. Fraud Confidence")
                    scatter_chart = alt.Chart(df.sample(min(1000, len(df)))).mark_circle(size=60, opacity=0.6).encode(
                        x=alt.X('Amount:Q', title='Transaction Amount'),
                        y=alt.Y('Confidence:Q', title='Fraud Confidence (%)'),
                        color=alt.Color('Prediction:N', 
                                       scale=alt.Scale(domain=[0, 1], 
                                                      range=['#63cdda', '#ff6b6b'] if theme == 'light' else ['#3d7ea6', '#ff6b6b']),
                                       legend=alt.Legend(title="Transaction Type")),
                        tooltip=['Amount', 'Confidence', 'Result']
                    ).properties(
                        width='container',
                        height=400,
                        title='Transaction Amount vs. Fraud Confidence'
                    ).interactive()
                    
                    st.altair_chart(scatter_chart, use_container_width=True)

        except Exception as e:
            st.markdown(f"""
            <div class='alert alert-danger'>
            <strong>❌ Error:</strong> {e}
            </div>
            """, unsafe_allow_html=True)

    # Previous uploads section
    st.divider()
    st.markdown("### 📁 Previous Uploads")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🔄 Show Last Upload", key="show_last_uploaded") and "last_uploaded" in st.session_state:
            st.session_state["show_last"] = True
    
    with col2:
        if os.path.exists(UPLOAD_DIR):
            files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith('.csv')]
            if files:
                selected_file = st.selectbox(
                    "Select a previous analysis", 
                    options=sorted(files, reverse=True),
                    key="previous_files",
                    help="View results from previous analyses"
                )
                if st.button("📂 Load Selected File"):
                    st.session_state["last_uploaded"] = os.path.join(UPLOAD_DIR, selected_file)
                    st.session_state["show_last"] = True
            else:
                st.info("No previous uploads found")
    
    if "show_last" in st.session_state and st.session_state["show_last"] and "last_uploaded" in st.session_state:
        try:
            last_uploaded_file = st.session_state["last_uploaded"]
            st.markdown("### 🔍 Previous Analysis Results")
            
            df_last = pd.read_csv(last_uploaded_file)
            
            if "Prediction" in df_last.columns:
                fraud_count = int(df_last['Prediction'].sum())
                legit_count = len(df_last) - fraud_count
                fraud_rate = fraud_count / len(df_last) * 100
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Total Transactions", f"{len(df_last):,}")
                
                with metric_col2:
                    st.metric("Fraud Detected", f"{fraud_count:,}")
                
                with metric_col3:
                    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
            
            # Show data preview
            st.dataframe(df_last.head(10), use_container_width=True)
            
            # Download button
            st.download_button(
                "📥 Download This Result",
                df_last.to_csv(index=False).encode('utf-8'),
                os.path.basename(last_uploaded_file),
                "text/csv"
            )
            
            # Clear button
            if st.button("❌ Close Preview"):
                st.session_state.pop("show_last", None)
                st.experimental_rerun()
                
        except Exception as e:
            st.error(f"⚠️ Failed to load file: {e}")
# --------------------- TAB 2: Feature Visualization ---------------------
with tab2:
    st.markdown("### 📊 Visualize Transaction Features")
    uploaded_viz = st.file_uploader("Upload CSV for Visualization", type=["csv"], key="viz")

    if uploaded_viz is not None:
        try:
            df_viz = pd.read_csv(uploaded_viz)

            required_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            missing_cols = [col for col in required_cols if col not in df_viz.columns]
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.stop()

            st.success("✅ File loaded successfully!")

            # Save uploaded file permanently
            filename = f"viz_uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            file_path = os.path.join(UPLOAD_DIR, filename)
            df_viz.to_csv(file_path, index=False)
            st.session_state["last_uploaded"] = file_path

            st.subheader("📉 2D PCA of the Transactions")
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df_viz[required_cols])

            fig, ax = plt.subplots()
            if "Class" in df_viz.columns:
                scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=df_viz["Class"], cmap="coolwarm", alpha=0.7)
                legend = ax.legend(*scatter.legend_elements(), title="Class")
                ax.add_artist(legend)
            else:
                ax.scatter(pca_result[:, 0], pca_result[:, 1], color="blue", alpha=0.5)
                ax.text(0.5, 0.9, "Note: 'Class' column missing — no fraud coloring", transform=ax.transAxes, ha='center', fontsize=9, color="gray")

            ax.set_title("PCA - 2D Projection")
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            st.pyplot(fig)

            # Feature correlation heatmap
            st.subheader("📊 Correlation Heatmap")
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            corr = df_viz.corr()
            sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax2)
            ax2.set_title("Correlation Heatmap of Features")
            st.pyplot(fig2)

            # Feature Importance (Optional)
            if 'Class' in df_viz.columns:
                st.subheader("📊 Feature Importance (Random Forest)")
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100)
                X = df_viz.drop(columns=["Class"])
                y = df_viz["Class"]
                model.fit(X, y)

                feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                feature_importances.plot(kind="bar", ax=ax3)
                ax3.set_title("Feature Importance")
                ax3.set_ylabel("Importance")
                st.pyplot(fig3)

            # Add Download Button for Results
            csv_data = df_viz.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Processed Result CSV", csv_data, "processed_visualization_results.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")

    # Show last uploaded file (from session state)
    if st.button("📁 Show Last Uploaded CSV", key="show_last_uploaded_tab3") and "last_uploaded" in st.session_state:
        try:
            last_uploaded_file = st.session_state["last_uploaded"]
            st.markdown("### 🔁 Last Uploaded CSV Preview")
            df_last = pd.read_csv(last_uploaded_file)
            st.dataframe(df_last.head())
        except Exception as e:
            st.error(f"⚠️ Failed to load last uploaded file: {e}")


# ---------------- Tab 3: Anomaly Detection ----------------
with tab3:
    st.markdown("### 🧠 Anomaly Detection")

    # File uploader
    uploaded_file_anomaly = st.file_uploader("Upload CSV for Anomaly Detection", type=["csv"], key="anomaly")

    # Save directory and path
    UPLOAD_DIR = "uploaded_files"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    LAST_FILE_PATH = os.path.join(UPLOAD_DIR, "last_uploaded_anomaly.csv")

    if uploaded_file_anomaly is not None:
        try:
            df_raw = pd.read_csv(uploaded_file_anomaly)

            # Validate required features
            required_features = model.feature_names_in_
            missing_features = set(required_features) - set(df_raw.columns)
            if missing_features:
                st.error(f"🚫 Missing features in CSV: {', '.join(missing_features)}")
                st.stop()

            # Prepare input
            df = df_raw[required_features]

            # Run model
            predictions = model.predict(df)
            prediction_probs = model.predict_proba(df)[:, 1]

            # Final result
            df_results = df_raw.copy()
            df_results["Prediction"] = predictions
            df_results["Confidence"] = prediction_probs * 100
            df_results["Result"] = df_results["Prediction"].map({0: "✅ Legit", 1: "🚨 Fraud"})

            # Save the result to file permanently
            df_results.to_csv(LAST_FILE_PATH, index=False)

            # Show top 5 anomalies
            df_sorted = df_results.sort_values(by="Confidence", ascending=False)
            st.markdown("#### 📊 Top 5 Most Anomalous Transactions")
            st.dataframe(df_sorted.head())

            # Bar chart (altair)
            import altair as alt
            top_anomalies = df_sorted.head()
            chart = alt.Chart(top_anomalies.reset_index()).mark_bar().encode(
                x='Confidence:Q',
                y=alt.Y('index:N', sort='-x'),
                color='Prediction:N',
                tooltip=['Result', 'Confidence']
            ).properties(title="Top 5 Anomalous Transactions (by Confidence)")
            st.altair_chart(chart, use_container_width=True)

            # Download full result
            st.download_button(
                label="📥 Download Full Result CSV",
                data=df_results.to_csv(index=False),
                file_name="anomaly_detection_results.csv",
                mime="text/csv",
            )

            # Summary
            st.success(f"🎯 Anomaly Detection complete! {df_results['Prediction'].sum()} fraud(s) found out of {len(df_results)} transactions.")

        except Exception as e:
            st.error(f"❌ Error: {e}")

    # 🔽 Show last uploaded file 
    if st.button("📁 Show Last Uploaded CSV ") and os.path.exists(LAST_FILE_PATH):
        try:
            st.markdown("### 🔁 Last Uploaded CSV Preview ")
            df_last = pd.read_csv(LAST_FILE_PATH)
            st.dataframe(df_last.head())
        except Exception as e:
            st.error(f"⚠️ Failed to load last uploaded file: {e}")

# ------------------ TAB 4: Model Details -----------

# ------------------ TAB 4: Model Details ---------------------- 
with tab4:
    # Model Overview
    st.markdown("### ℹ️ Model Details")
    st.markdown("""
    - **Model Type**: Random Forest Classifier  
    - **Dataset**: Credit Card Fraud Detection dataset (Kaggle)  
    - **Accuracy**: 99.8%  
    - **Objective**: Detect fraudulent transactions based on historical data  
    - **Preprocessing**: Normalization and PCA used  
    """)

    # --- Model Performance Chart ---
    st.markdown("#### 📊 Model Performance Comparison")

    model_names = ['Random Forest', 'Logistic Regression', 'SVM']
    accuracies = [99.8, 95.4, 98.5]

    sns.set(style="whitegrid")
    fig1, ax1 = plt.subplots(figsize=(4.5, 2))
    sns.barplot(x=model_names, y=accuracies, palette="Blues_d", ax=ax1)

    ax1.set_xlabel('Model', fontsize=9)
    ax1.set_ylabel('Accuracy (%)', fontsize=9)
    ax1.set_title('Accuracy Comparison', fontsize=10)
    ax1.tick_params(axis='x', labelsize=8, rotation=10)
    ax1.tick_params(axis='y', labelsize=8)

    st.pyplot(fig1)

    # --- Confusion Matrix ---
    st.markdown("#### 🔍 Confusion Matrix (Example)")

    y_true = [0, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    y_pred = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1]

    cm = confusion_matrix(y_true, y_pred)

    fig2, ax2 = plt.subplots(figsize=(3.5, 2.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'], ax=ax2)

    ax2.set_xlabel("Predicted Label", fontsize=9)
    ax2.set_ylabel("True Label", fontsize=9)
    ax2.set_title("Confusion Matrix", fontsize=10)
    ax2.tick_params(axis='x', labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)

    st.pyplot(fig2)

    # --- Explanation for Users ---
    st.markdown("""
    **Legend:**  
    - **TP**: Correctly predicted fraud  
    - **TN**: Correctly predicted non-fraud  
    - **FP**: Non-fraud predicted as fraud  
    - **FN**: Fraud predicted as non-fraud  
    """)

# ---- FOOTER ----
st.markdown("<div class='footer'>Made by Kishori Kumari | MITS Gwalior</div>", unsafe_allow_html=True)

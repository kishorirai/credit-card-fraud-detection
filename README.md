# **Credit Card Fraud Detection**

This project aims to build a **Credit Card Fraud Detection** system using machine learning. The model is trained on historical credit card transaction data to predict whether a transaction is fraudulent or legitimate. The app is built using **Streamlit**, which provides an interactive web interface for easy deployment and user interaction.

## **Technologies Used**

- **Python**: Programming language used for the model and app.
- **Streamlit**: Framework for creating the web app interface.
- **Scikit-Learn**: Machine learning library for training the model.
- **Pandas**: Data manipulation and analysis.
- **Pickle**: Serialization of the trained machine learning model.
- **Matplotlib/Seaborn**: For data visualization (if used).

## **Features**

- Upload a transaction dataset (in CSV format) for fraud detection.
- Get predictions on whether a transaction is fraudulent or not based on the trained model.
- View the accuracy and model evaluation metrics.

## **How to Run the App Locally**

1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/credit-card-fraud-detection.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

4. Visit `http://localhost:8501` in your browser to interact with the app.

## **Deployment**

The app is hosted on **Streamlit Cloud**, which makes it accessible online. You can access the deployed app at the following URL:
https://credit-card-fraud-detection-p9onhhkshnncs4xnykspa6.streamlit.app/

## **Dataset**

The dataset used for training the model is available in the repository in zip file, credit_card_fraud_data.csv`. It contain features related to transactions, such as transaction amount, time, and anonymized features. You can use this dataset to test the model.

## **Model**

The trained machine learning model is saved using the **Pickle** format and can be loaded into the app to make predictions. Based on the input data, the model can classify transactions as either **fraudulent** or **legitimate**.

## **Author**

This project was developed by Kishori Kumari, a student at Madhav Institute of Technology and Science (MITS Gwalior).

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

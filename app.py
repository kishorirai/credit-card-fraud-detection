        # Drop the 'Class' column if it exists
        if 'Class' in df.columns:
            df = df.drop(columns=['Class'])

        # Check if all required columns are present after dropping
        if all(col in df.columns for col in required_columns):
            st.write("🔍 Preview of Uploaded Data:")
            st.dataframe(df.head())

            predictions = model.predict(df)
            df["Prediction"] = predictions
            df["Result"] = df["Prediction"].map({0: "✅ Legit", 1: "🚨 Fraud"})

            st.success("Predictions completed!")
            st.dataframe(df[["Prediction", "Result"]])

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")
        else:
            st.error(f"Error: Missing required columns. Expected columns: {', '.join(required_columns)}")

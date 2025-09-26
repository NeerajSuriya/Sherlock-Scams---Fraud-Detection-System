import streamlit  as st
import time
import pickle 
import numpy as np
import pandas as pd
from PIL import Image

model_path = "models/LogisticRegression_model.pkl"
with open(model_path,"rb") as model_file:
    model = pickle.load(model_file)

st.title("Sherlock Scams - Fraud Detection System")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
    try:
        if st.button("Check for Fraudulent Transactions"):   
            st.write("Processing...")
            expected_features = model.feature_names_in_
            with st.spinner("Processing transaction..."):
                progress_bar = st.progress(0)
                for i in range(5):  # Simulate loading time
                    time.sleep(1)
                    progress_bar.progress((i + 1) * 20)

            # Drop target column if present
            if "Class" in dataframe.columns:
                dataframe = dataframe.drop(columns=["Class"])

            # Select columns in correct order
            dataframe = dataframe[expected_features]

            predictions = model.predict(dataframe)
            fraudulent_transactions = dataframe[predictions == 1]
            
            st.write("Fraudulent Transactions Summary:")
            st.write(f"Total Transactions: {len(dataframe)}")
            st.write(f"Number of Fraudulent Transactions: {len(fraudulent_transactions)}")
            
            if len(fraudulent_transactions) > 0:
                st.write("Details of Fraudulent Transactions:")
                st.write(fraudulent_transactions)
            else:
                st.write("No fraudulent transactions were detected.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
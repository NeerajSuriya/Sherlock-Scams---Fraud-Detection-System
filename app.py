import streamlit as st
import time
import pickle
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Load Model
# ------------------------------
model_path = "models/LogisticRegression_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Sherlock Scams 🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

body {
    font-family: 'Poppins', sans-serif;
    color: #ffffff;
}

/* Background wallpaper */
[data-testid="stAppViewContainer"]{
    background-size: cover;
    background-position: center;
}

/* Hero Section */
.hero {
    padding: 60px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 30px;
    background: rgba(0,0,0,0.6);
}
.hero h1 {
    font-size: 60px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 10px;
}
.hero p {
    font-size: 22px;
    color: #f0f0f0;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.85);
    color: #000;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    text-align:center;
    margin-bottom: 20px;
}
.kpi-number {
    font-size: 36px;
    font-weight: 700;
}

/* Buttons */
.stButton>button {
    background-color: #4CAF50;
    color:white;
    height:50px;
    width:100%;
    border-radius:10px;
    font-weight:600;
}

/* Table Hover */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.9);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Hero Section
# ------------------------------
st.markdown("""
<div class="hero">
    <h1>🕵️ Sherlock Scams</h1>
    <p>Detect fraudulent transactions instantly with our smart AI model.</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "Upload a CSV of transactions. Our Logistic Regression model flags fraud "
    "based on patterns learned from historical data."
)

# ------------------------------
# File Upload
# ------------------------------
uploaded_file = st.file_uploader("📂 Drag & drop your transaction CSV here", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.markdown("### 📄 Uploaded Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    if st.button("🚀 Analyze Transactions"):
        expected_features = model.feature_names_in_

        # ------------------------------
        # Loading Simulation
        # ------------------------------
        with st.spinner("Analyzing transactions..."):
            progress_bar = st.progress(0)
            for i in range(5):
                time.sleep(0.5)
                progress_bar.progress((i+1)*20)

        # ------------------------------
        # Prepare Data
        # ------------------------------
        if "Class" in df.columns:
            df = df.drop(columns=["Class"])
        df = df[expected_features]

        scaler = StandardScaler()
        df[expected_features] = scaler.fit_transform(df[expected_features])

        # ------------------------------
        # Predictions
        # ------------------------------
        df["Prediction"] = model.predict(df)
        fraud_df = df[df["Prediction"] == 1]

        total_txn = len(df)
        fraud_count = len(fraud_df)
        fraud_percent = (fraud_count / total_txn) * 100 if total_txn > 0 else 0

        # ------------------------------
        # KPI Section (Columns)
        # ------------------------------
        st.markdown("### 📊 Key Metrics")
        col1, col2, col3 = st.columns(3)

        col1.markdown(f"<div class='card'>🧾<br>Total Transactions<br><span class='kpi-number'>{total_txn}</span></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='card'>🚨<br>Fraudulent<br><span class='kpi-number' style='color:red'>{fraud_count}</span></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='card'>📉<br>Fraud %<br><span class='kpi-number' style='color:green'>{fraud_percent:.2f}%</span></div>", unsafe_allow_html=True)

        # ------------------------------
        # Charts Section (Columns)
        # ------------------------------
        st.markdown("### 📈 Visualizations")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            summary = pd.DataFrame({
                "Type": ["Legitimate", "Fraudulent"],
                "Count": [total_txn - fraud_count, fraud_count]
            })
            pie_chart = alt.Chart(summary).mark_arc(innerRadius=60).encode(
                theta="Count",
                color="Type",
                tooltip=["Type", "Count"]
            )
            st.altair_chart(pie_chart, use_container_width=True)

        with chart_col2:
            line_chart = alt.Chart(df.reset_index()).mark_line(point=True).encode(
                x="index",
                y="Prediction",
                tooltip=["index", "Prediction"]
            ).properties(height=300)
            st.altair_chart(line_chart, use_container_width=True)

        # ------------------------------
        # Table Section (Full Width)
        # ------------------------------
        st.markdown("### 🚨 Fraudulent Transactions Details")
        if fraud_count > 0:
            def highlight_fraud(row):
                return ['background-color: #FFB3B3; font-weight: bold;' if row["Prediction"]==1 else '' for _ in row]
            st.dataframe(fraud_df.style.apply(highlight_fraud, axis=1), use_container_width=True)
        else:
            st.success("✅ No fraudulent transactions were detected.")

        # ------------------------------
        # Detailed Analysis Section
        # ------------------------------
        with st.expander("🔍 Full Data Analysis"):
            st.write(df.describe())
            st.write("Correlation Matrix:")
            st.dataframe(df.corr(), use_container_width=True)

# Sherlock-Scams – Fraud Detection System

## 📌 Context
Credit card companies need to detect fraudulent transactions to prevent customers from being charged for unauthorized purchases.  
Fraud detection systems must balance **speed, accuracy, and scalability** to minimize losses and protect user trust.

---

## 📊 Dataset
- **Source**: Transactions made by European cardholders in **September 2013**.  
- **Collected by**: Worldline and the Machine Learning Group (MLG) of Université Libre de Bruxelles (ULB).  
- **Duration**: Two days of transactions.  
- **Total Records**: 284,807 transactions.  
- **Fraud Cases**: 492 (≈ 0.172% of all transactions).  

🔗 More details: [MLG Website](http://mlg.ulb.ac.be) | [ResearchGate Project](https://www.researchgate.net/project/Fraud-detection-5)

---

## ⚙️ Features
- **Numerical Input Variables**: Transformed using **Principal Component Analysis (PCA)** to protect confidentiality.  
- **Time**: Seconds elapsed between each transaction and the first transaction.  
- **Amount**: Transaction amount, suitable for cost-sensitive learning.  
- **Class (Target Variable)**:  
  - `1` → Fraudulent transaction  
  - `0` → Genuine transaction  

---

## 🚨 Challenges
- **Imbalanced Data**: Only 0.172% are fraud cases → standard accuracy is misleading.  
- **Feature Confidentiality**: PCA-transformed features reduce interpretability.  
- **Real-Time Constraints**: Fraud detection must be fast enough to block suspicious transactions instantly.  

---

## 📈 Evaluation Metrics
- **Preferred**: Area Under the Precision-Recall Curve (AUPRC).  
- **Additional Metrics**: ROC AUC, Recall (catching frauds), Precision (reducing false alarms).  
- ❌ **Confusion Matrix Accuracy** is not meaningful due to imbalance.  

---

## 🛠️ Recommendations
- Apply **resampling techniques**: SMOTE, oversampling, undersampling.  
- Use **ensemble models**: Random Forest, XGBoost, LightGBM.  
- Implement a **pipeline** with scaling, encoding, and cross-validation.  
- Explore **anomaly detection**: Isolation Forests, Autoencoders.  

---

## 💡 Applications
- **Banking & Finance**: Flagging

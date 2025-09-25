import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import pickle

def load_processed_data(file_path):
    return pd.read_csv(file_path)

def prepare_training_data(df):
    X = df.drop(columns='Class')
    y = df['Class']

    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    train_data = X_train_orig.copy()
    train_data['Class'] = y_train_orig

    majority_class = train_data[train_data['Class'] == 0]
    minority_class = train_data[train_data['Class'] == 1]

    majority_downsampled = resample(majority_class, 
                                    replace=False, 
                                    n_samples=len(minority_class), 
                                    random_state=42)

    downsampled_train_data = pd.concat([majority_downsampled, minority_class])

    X_train_downsampled = downsampled_train_data.drop(columns='Class')
    y_train_downsampled = downsampled_train_data['Class']

    return X_train_downsampled, y_train_downsampled, X_test_orig, y_test_orig

def train_logistic_regression(X_train, y_train):
    
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    return log_reg

def save_model(model, directory, filename):

    if not os.path.exists(directory):
        os.makedirs(directory)
    model_filepath = os.path.join(directory, filename)
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_filepath}")

def save_classification_report(y_true, y_pred, file_path):

    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues', fmt='.2f')
    plt.title('Classification Report')
    plt.savefig(file_path)
    print(f"Classification report saved to {file_path}")

def plot_confusion_matrix(y_true, y_pred):

    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Fraud', 'Fraud'], 
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

processed_data_path = 'data/processed/processed_data.csv'
df = load_processed_data(processed_data_path)
print("Data Loaded!")

X_train_downsampled, y_train_downsampled, X_test_orig, y_test_orig = prepare_training_data(df)

log_reg = train_logistic_regression(X_train_downsampled, y_train_downsampled)
print("Model Trained!")

y_pred = log_reg.predict(X_test_orig)

plot_confusion_matrix(y_test_orig, y_pred)

save_model(log_reg, 'models', 'logistic_regression_model.pkl')

classification_report_path = 'artifacts/classification_report.jpeg'
save_classification_report(y_test_orig, y_pred, classification_report_path)
print("Report Saved!")
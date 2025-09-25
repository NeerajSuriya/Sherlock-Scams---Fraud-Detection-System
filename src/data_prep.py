import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomUnderSampler
import os

def prepocess_data(df):
    X = df.drop('Class', axis = 1)
    y = df['Class']

    rus = RandomUnderSampler(random_state = 42)

    X_resampled, y_resampled = rus.fit_resample(X,y)

    downsampled_df = pd.concat([pd.DataFrame(X_resampled, columns = X.columns), pd.DataFrame(y_resampled, columns=['Class'])], axis = 1)

    return downsampled_df

def save_processed_data(df, file_path):
    df.to_csv(file_path, index = False)
    print(f"Processed data saved to {file_path}")

def plot_heatmap(df, file_path):
    file_path = 'artifacts/heatmap.jpeg'
    plt.figure(figsize=(16,9))
    sns.heatmap(df.corr(), annot = True)
    plt.savefig(file_path)
    print(f"Heatmap saved to {file_path}")

data_path = 'data/raw_data/creditcard.csv'
df = pd.read_csv(data_path)

downsampled_df = prepocess_data(df)

processed_data_path = 'data/processed/processed_data.csv'
save_processed_data(df, processed_data_path)

heatmap_path = 'artificial/heatmap.jpeg'
plot_heatmap(downsampled_df, heatmap_path)

import streamlit  as st
import time
import pickle 
import numpy as np
import pandas as pd
from PIL import Image

model_path = "model/LogisticRegression_model.pkl"
with open(model_path,"rb") as model_file:
    model = pickle.load(model_file)

st.title("Sherlock Scams - Fraud Detection System")
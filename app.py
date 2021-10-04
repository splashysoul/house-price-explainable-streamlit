import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import streamlit as st
import shap


boston_dataset = load_boston()
dataset = pd.DataFrame(boston_dataset.data, columns = boston_dataset.feature_names)
dataset['MEDV'] = boston_dataset.target

X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 25)

st.title('Explainable Machine Learning')

task = st.sidebar.selectbox('Select task', ('Boston House', 'Heart Disease'))

if task == 'Boston House':
  
  steps = st.sidebar.selectbox('Select ', ('Data Visualization', 'Model Prediction and Explain'))
  
  if steps == 'Data Visualization':
    corr = dataset.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr, cmap='RdBu', annot=True, fmt=".2f")
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    st.write(fig)
  
  if steps == 'Model Prediction and Explain':
    model = st.sidebar.selectbox('Select model', ('linear regression', 'XGBoost', ''))

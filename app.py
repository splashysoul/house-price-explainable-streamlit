import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence
from sklearn.inspection import permutation_importance
import xgboost
import lightGBM

import streamlit as st
import shap
import lime




st.title('Explainable Machine Learning')

task = st.sidebar.selectbox('Select task', ('Boston House', 'Heart Disease'))

if task == 'Boston House':
  boston_dataset = load_boston()
  dataset = pd.DataFrame(boston_dataset.data, columns = boston_dataset.feature_names)
  dataset['MEDV'] = boston_dataset.target
  X = dataset.iloc[:, 0:13].values
  y = dataset.iloc[:, 13].values.reshape(-1,1)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 25)

if task == 'Boston House':
  
  steps = st.sidebar.selectbox('Select ', ('Data Visualization', 'Model Prediction and Explain'))
  
  if steps == 'Data Visualization':
    corr = dataset.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr, cmap='RdBu', annot=True, fmt=".2f")
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    st.write(fig)
    sns.pairplot(dataset)
    plt.show()
  
  if steps == 'Model Prediction and Explain':
    model = st.sidebar.selectbox('Select model', ('linear regression', 'XGBoost', ''))
    if model == 'linear regression':
      regressor_linear = LinearRegression()
      regressor_linear.fit(X_train, y_train)
      cv_linear = cross_val_score(estimator = regressor_linear, X = X_train, y = y_train, cv = 10)
      y_pred_linear_train = regressor_linear.predict(X_train)
      r2_score_linear_train = r2_score(y_train, y_pred_linear_train)
      y_pred_linear_test = regressor_linear.predict(X_test)
      r2_score_linear_test = r2_score(y_test, y_pred_linear_test)
      rmse_linear = (np.sqrt(mean_squared_error(y_test, y_pred_linear_test)))
      st.subheader('Weight Plot')
     if model == 'XGBoost':
      model = xgboost.XGBRegressor().fit(X_train, y_train)
      explainer = shap.Explainer(model)
      shap_values = explainer(X)
      
      shap.plots.waterfall(shap_values[0])

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])

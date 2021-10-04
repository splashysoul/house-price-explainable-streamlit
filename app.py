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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence
from sklearn.inspection import permutation_importance
import xgboost
#import lightGBM

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
  col1, col2 = st.sidebar.beta_columns(2)
  CRIM = col1.number_input(label='CRIM', min_value=0, max_value=100,format='%i',step=1, value=20)
  ZN = col2.number_input(label='ZN', min_value=0.0, max_value=1.0,format= '%.6f',step=0.000001, value=0.5)
  INDUS = col1.number_input(label='energy', min_value=0.0, max_value=1.0,format= '%.6f',step=0.000001, value=0.5)
  CHAS = col2.number_input(label='key', min_value=0, max_value=11,format= '%i', value=0)
  NOX = col1.number_input(label='loudness', min_value=-50.0, max_value=2.0,format= '%.6f',step=0.000001, value=-6.0)
  RM = col2.number_input(label='mode', min_value=0, max_value=1,format= '%i',step=1, value=0)
  AGE = col1.number_input(label='speechiness', min_value=0.0, max_value=1.0,format= '%.6f',step=0.000001, value=0.5)
  DIS = col2.number_input(label='acousticness', min_value=0.0, max_value=1.00,format= '%.6f',step=0.000001, value=0.5)
  RAD = col1.number_input(label='instrumentalness', min_value=0.0, max_value=1.0,format= '%.6f',step=0.000001, value=0.5)
  TAX = col2.number_input(label='liveness', min_value=0.0, max_value=1.0,format= '%.6f',step=0.000001, value=0.5)
  PTRATIO = col1.number_input(label='valence', min_value=0.0, max_value=1.0,format= '%.6f',step=0.000001, value=0.5)
  B = col2.number_input(label='tempo', min_value=0.0, max_value=300.0,format= '%.6f',step=0.000001, value=120.0)
  LSTAT	 = col1.number_input(label='duration_ms', min_value=0, max_value=999999,format= '%i',step=1, value=225000)
  
  input_list = [CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTA]
  df_input = pd.DataFrame([input_list], columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])


  
  if steps == 'Data Visualization':
    corr = dataset.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr, cmap='RdBu', annot=True, fmt=".2f")
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    st.pyplot(fig)
    fig = sns.pairplot(dataset)
    st.pyplot(fig)
    
  
  if steps == 'Model Prediction and Explain':
    model = st.sidebar.selectbox('Select model', ('Linear Regression', 'XGBoost', 'Decision Tree', 'Random Forest'))
    
    if model == 'Linear Regression':
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
      shap_values = explainer(X_train)
      input_shap_values = explainer.shap_values(df_input)
      st.subheader('Force plot')
      force_plot = shap.force_plot(explainer.expected_value[np.argmax(input_preds_proba)],
                    input_shap_values[np.argmax(input_preds_proba)],
                    eval_set_features,
                    matplotlib=True,
                    show=False)
      st.pyplot(force_plot)
      
      # visualize the first prediction's explanation
      #shap.plots.waterfall(shap_values[0])
      force_plot_all = shap.plots.force(shap_values)
      st.pyplot(force_plot_all)
    
    if model == 'Decision Tree':
      regressor_dt = DecisionTreeRegressor(random_state = 0)
      regressor_dt.fit(X_train, y_train)
      st.subheader('partial_dependence')
      st.subheader('SHAP')
      st.subheader('LIME')
      
    if model == 'Random Forest' :
      regressor_rf = RandomForestRegressor(n_estimators = 500, random_state = 0)
      regressor_rf.fit(X_train, y_train.ravel())
      st.subheader('partial_dependence')
      st.subheader('SHAP')
      st.subheader('LIME')
      
    

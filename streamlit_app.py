import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt, log
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import streamlit as st

@st.cache
def load_data():
    # Read the data
    df = pd.read_csv("data_inr.csv")
    df['Name'] = pd.to_datetime(df['Name'])
    df.set_index('Name', inplace=True)
    return df

def main():
    st.title("Gold Price Analysis")

    # Load the data
    df = load_data()

    # Drop rows with missing values
    df.dropna(inplace=True)
    
 # Perform log transformation
    df['log_gold'] = np.log(df['Indian rupee'])

    # Check stationarity using ADF test
    result_of_adfuller = adfuller(df['log_gold'])
    p_value = result_of_adfuller[1]
    if p_value < 0.05:
        st.write("Time series is stationary")
    else:
        st.write("Time series is non-stationary")

    # Differencing of order 1
    df['diff_gold'] = df['log_gold'].diff()
    df.dropna(inplace=True)

    # Define exploratory variables
    df['S_1'] = df['diff_gold'].shift(1).rolling(window=3).mean()
    df['S_2'] = df['diff_gold'].shift(1).rolling(window=12).mean()

    # Split into train and test
    X = df[['S_1', 'S_2']]
    y = df['diff_gold']
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Perform linear regression
    linear = LinearRegression().fit(X_train, y_train)
    predicted_price = linear.predict(X_test)

    # Calculate R square and RMSE
    r2_score_val = r2_score(y_test, predicted_price)
    rmse_val = sqrt(mean_squared_error(y_test, predicted_price))
    st.write("R square for regression:", r2_score_val)
    st.write("RMSE:", rmse_val)

    # SARIMAX model
    mod = sm.tsa.statespace.SARIMAX(df['diff_gold'], order=(2, 1, 2), seasonal_order=(2, 1, 2, 12))
    results = mod.fit()
    df['sarimax_predict'] = results.predict()

    # Plot actual and predicted values
    fig, ax = plt.subplots()
    ax.plot(df.index, df['diff_gold'], label='Actual')
    ax.plot(df.index, df['sarimax_predict'], label='Predicted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Gold Price Difference')
    ax.set_title('Actual vs Predicted Gold Price Difference')
    ax.legend()
    st.pyplot(fig)
   # Check model performance
    rmse_sarimax = sqrt(mean_squared_error(df['diff_gold'], df['sarimax_predict']))
    r2_score_sarimax = r2_score(df['diff_gold'], df['sarimax_predict'])
    st.write("SARIMAX RMSE:", rmse_sarimax)
    st.write("SARIMAX R2 SCORE:", r2_score_sarimax)

if _name_ == '_main_':
    main()


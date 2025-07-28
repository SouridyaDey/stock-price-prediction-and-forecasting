# 📈 Stock Price Prediction and Forecasting

This project aims to predict the **next day's closing price** of Apple Inc. (AAPL) stock using advanced time series forecasting techniques. After thorough exploratory data analysis (EDA) and experimentation with multiple statistical and machine learning models, a **Stacked LSTM (Long Short-Term Memory)** neural network was selected for final deployment due to its superior performance in capturing temporal dependencies.

---

## 🧠 Models Explored

- **ARIMA (AutoRegressive Integrated Moving Average)**
- **Exponential Smoothing (ETS)**
- **Prophet (by Meta)**
- **Stacked LSTM** ✅ *(Selected Final Model)*

After comparing all models based on **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)**, **Stacked LSTM** outperformed the rest.

---

## 🔍 Project Workflow

### 1. **Data Collection**
- Stock data fetched using [`yfinance`](https://pypi.org/project/yfinance/) for AAPL from 2010 to present.

### 2. **Exploratory Data Analysis (EDA)**
- Visual trend analysis, seasonality inspection, and statistical summary.
- Stationarity tests (ADF Test) and decomposition.

### 3. **Modeling**
- ARIMA and ETS: Applied using `statsmodels`.
- Prophet: Trained with custom seasonality.
- LSTM: 
  - Input sequence: last 300 days
  - Layers: 3 stacked LSTM layers with Dense output
  - Loss: Mean Squared Error
  - Optimizer: Adam
  - Evaluation: RMSE & MAE on test set

### 4. **Forecasting**
- Model forecasts **90 future business days** using a recursive sequence approach.
- Final results are inverse-transformed to original scale using a saved `MinMaxScaler`.

### 5. **Deployment**
- Flask web app developed to display forecasted prices.
- Model and scaler serialized using `pickle`.
- Deployed on [Render](https://render.com) using `gunicorn` as production WSGI server.

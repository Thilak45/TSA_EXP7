# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 29-09-2025



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Load the gold price dataset
data = pd.read_csv("C:/Users/admin/Downloads/archive (2)/Gold_price_2025 new.csv")

# Automatically detect the 'close' price column
close_cols = [col for col in data.columns if 'close' in col.lower()]
if close_cols:
    close_col = close_cols[0]
else:
    raise Exception('No close price column detected.')

# Ensure proper numeric conversion (strip commas or odd characters if needed)
time_series = pd.to_numeric(data[close_col].astype(str).str.replace(',', '').str.strip(), errors='coerce')

# ADF test for stationarity
result = adfuller(time_series.dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])

# Split to train and test
train_size = int(len(time_series) * 0.8)
train, test = time_series.iloc[:train_size], time_series.iloc[train_size:]

# Plot ACF and PACF
plot_acf(time_series.dropna(), lags=30)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plot_pacf(time_series.dropna(), lags=30)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# Fit AutoRegressive model (lag=5 as a starting point based on analysis)
model = AutoReg(train, lags=5, old_names=False).fit()
print(model.summary())

# Predict
preds = model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# Calculate Mean Squared Error
error = mean_squared_error(test, preds)
print("Mean Squared Error:", error)

# Plot actual vs predicted
plt.figure(figsize=(10,5))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, preds, label='Predicted', color='red')
plt.legend()
plt.title("Gold Price Time Series: Actual vs Predicted")
plt.show()

```
### OUTPUT:

<img width="1317" height="618" alt="image" src="https://github.com/user-attachments/assets/f9d9a58a-046e-4d03-ab9d-4f5fb0f3c187" />

<img width="870" height="552" alt="image" src="https://github.com/user-attachments/assets/9ae4c7c1-7d62-4688-92c4-033e0649c521" />

<img width="905" height="633" alt="image" src="https://github.com/user-attachments/assets/52859ad7-db21-4f81-aaf3-f87a73da2c6b" />


FINIAL PREDICTION

<img width="1073" height="572" alt="image" src="https://github.com/user-attachments/assets/9087d5d3-7106-44ec-b58f-4ed5e44268a2" />


### RESULT:
Thus we have successfully implemented the auto regression function using python.

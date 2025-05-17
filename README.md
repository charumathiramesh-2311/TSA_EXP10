# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```c
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Step 1: Load and preprocess dataset
data = pd.read_csv('IOT-temp.csv')  # Replace with correct path
data['noted_date'] = pd.to_datetime(data['noted_date'], format='%d-%m-%Y %H:%M')
data.set_index('noted_date', inplace=True)

# Step 2: Resample to daily average temperature
daily_temp = data['temp'].resample('D').mean().dropna()
plt.plot(data.index, data['temp'])
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature Time Series')
plt.show()


# Step 3: Check for stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print(f'   {key}: {value}')

check_stationarity(daily_temp)

# Step 4: Plot ACF and PACF
plot_acf(daily_temp)
plt.title("ACF Plot")
plt.show()

plot_pacf(daily_temp)
plt.title("PACF Plot")
plt.show()

# Step 5: Split data into train and test sets
train_size = int(len(daily_temp) * 0.8)
train, test = daily_temp[:train_size], daily_temp[train_size:]

# Step 6: Fit SARIMA model (adjust parameters as needed)
model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,7))  # weekly seasonality
result = model.fit()

# Step 7: Forecast
predictions = result.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# Step 8: Evaluate model
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('Root Mean Squared Error (RMSE):', rmse)

# Step 9: Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, label='Predicted', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('SARIMA Model Predictions on IoT Temperature Data')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


```
### OUTPUT:

![image](https://github.com/user-attachments/assets/48c57e90-51d0-486c-b62b-195c4cacb8db)


![image](https://github.com/user-attachments/assets/ac200796-c47e-4c17-8b7e-ed04ed85b19b)

![image](https://github.com/user-attachments/assets/e414b2bc-c736-4784-968d-54d894f3eb72)

![image](https://github.com/user-attachments/assets/891a7dca-c07a-4db8-999e-1f0709c27ff8)


![image](https://github.com/user-attachments/assets/4de5c7c7-275b-4a2f-9f5b-74dfdc922901)


### RESULT:
Thus the program run successfully based on the SARIMA model.

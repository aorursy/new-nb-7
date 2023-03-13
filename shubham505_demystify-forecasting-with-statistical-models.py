# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
 # we are going to use some statical models as well as our traditional linear regression model, so we will use sklearn and statsmodel to import those
#for basic operations
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from pandas_profiling import ProfileReport

#for vizualizations
from matplotlib import pyplot as plt
import seaborn as sn

#Sklearn imports
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

#for all statical modles
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import warnings
warnings.filterwarnings('ignore')
sales_data = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/train.csv')
sales_data
#lets generate profile reports from a pandas DataFrame , for quick data analysis
profile = ProfileReport(sales_data)
profile
#no missing values in the dataset.
# Filter records for store 1 and item 1 -> to be able to scale to other items in the future
sales_data = sales_data[sales_data['store'] == 1]
sales_data = sales_data[sales_data['item'] == 1]

sales_data['date'] = pd.to_datetime(sales_data['date'], format='%Y-%m-%d') # convert date column to datatime object

# Create Date-related Features to be used for EDA and Supervised ML: Regression
sales_data['year'] = sales_data['date'].dt.year
sales_data['month'] = sales_data['date'].dt.month
sales_data['day'] = sales_data['date'].dt.day
sales_data['weekday'] = sales_data['date'].dt.weekday
sales_data['weekday'] = np.where(sales_data.weekday == 0, 7, sales_data.weekday)

# Split the series to predict the last 3 months of 2017
temp_df = sales_data.set_index('date')
train_df = temp_df.loc[:'2017-09-30'].reset_index(drop=False)                         
test_df = temp_df.loc['2017-10-01':].reset_index(drop=False)

train_df.head()
test_df.head()
monthly_agg = sales_data.groupby('month')['sales'].sum().reset_index()
fig, axs = plt.subplots(nrows=2, figsize=(9,7))
sn.boxplot(x='month', y='sales', data=sales_data, ax=axs[0])
_ = sn.lineplot(x='month', y='sales', data=monthly_agg, ax=axs[1])
plot = sn.boxplot(x='weekday', y='sales', data=sales_data)
_ = plot.set(title='Weekly distribution')
yearly_agg = sales_data.groupby('year')['sales'].sum().reset_index()
fig, axs = plt.subplots(nrows=2, figsize=(9,7))
sn.boxplot(x='year', y='sales', data=sales_data, ax=axs[0])
_ = sn.lineplot(x='year', y='sales', data=yearly_agg, ax=axs[1])
plot = sn.lineplot(x='date', y='sales', data=sales_data)
_ = plot.set(title='Sales for Store 1, Item 1 over the years')
# subtract 1 year from test data
dates = (test_df['date'] - np.timedelta64(1, 'Y') + \
        np.timedelta64(1, 'D')).values.astype('datetime64[D]') 
seasonal_naive_sales = train_df[train_df['date'].astype('datetime64[D]').isin(dates)]['sales'] 

# make a copy of the test_df and make naive predictions for the last 3 months of 2017
sn_pred_df = test_df.copy().drop('sales', axis=1)
sn_pred_df['seasonal_naive_sales'] = pd.DataFrame(seasonal_naive_sales).set_index(test_df.index)
sn_pred_df.head()
plt.figure(figsize=(14,7))
plt.plot(train_df['date'], train_df['sales'], label='Train')
plt.plot(test_df['date'], test_df['sales'], label='Test')
plt.plot(sn_pred_df['date'], sn_pred_df['seasonal_naive_sales'], label='Forecast - Seasonal Naive')
plt.legend(loc='best')
plt.xlabel('date')
plt.ylabel('sales')
plt.title('Forecasts using Baseline Model: Seasonal Naive')
plt.show()
errors_df = pd.merge(test_df, sn_pred_df, on='date')
errors_df = errors_df[['date', 'sales', 'seasonal_naive_sales']]
errors_df = pd.merge(test_df, sn_pred_df, on='date')
errors_df = errors_df[['date', 'sales', 'seasonal_naive_sales']]
errors_df['errors'] = test_df['sales'] - sn_pred_df['seasonal_naive_sales']
errors_df.insert(0, 'model', 'Seasonal Naive') 

def mae(err):
    return np.mean(np.abs(err))

def rmse(err):
    return np.sqrt(np.mean(err ** 2))

def mape(err, sales=errors_df['sales']):
    return np.sum(np.abs(err))/np.sum(sales) * 100

result_df = errors_df.groupby('model').agg(total_sales=('sales', 'sum'),
                                           total_sn_pred_sales=('seasonal_naive_sales', 'sum'),
                                           overall_error=('errors', 'sum'),
                                           MAE=('errors', mae), 
                                           RMSE=('errors', rmse), 
                                           MAPE=('errors', mape))
    

plt.figure(figsize=(14,7))
plt.plot(errors_df['date'], np.abs(errors_df['errors']), label='errors')
plt.plot(errors_df['date'], errors_df['sales'], label='actual sales')
plt.plot(errors_df['date'], errors_df['seasonal_naive_sales'], label='forecast')
plt.legend(loc='best')
plt.xlabel('date')
plt.ylabel('sales')
plt.title('Seasonal Naive forecasts with actual sales and errors')
plt.show()

result_df
ts_decomp_df = train_df.set_index('date') # set date as index
ts_decomp_df['sales'] = ts_decomp_df['sales'].astype(float)
ts_decomp_df.head()
result = seasonal_decompose(ts_decomp_df['sales'], model='additive', freq=365)
fig = plt.figure()  
fig = result.plot()  
fig.set_size_inches(14, 12)
hw_train_df = train_df[['date', 'sales']].set_index('date')
hw_test_df = test_df[['date', 'sales']].set_index('date')

# Apply Triple Exponential Smoothing

hw_model_1 = ExponentialSmoothing(hw_train_df, trend='add', seasonal='add', seasonal_periods=12)
hw_fit_1 = hw_model_1.fit(use_boxcox=False, remove_bias=False)
pred_fit_1 = pd.Series(hw_fit_1.predict(start=hw_test_df.index[0], end=hw_test_df.index[-1]), 
                       name='pred_sales').reset_index()

hw_model_2 = ExponentialSmoothing(hw_train_df, trend='add', seasonal='add', seasonal_periods=12, damped=True)
hw_fit_2 = hw_model_2.fit(use_boxcox=False, remove_bias=False)
pred_fit_2 = pd.Series(hw_fit_2.predict(start=hw_test_df.index[0], end=hw_test_df.index[-1]), 
                       name='pred_sales').reset_index()
print('Forecasts made, ready for evaluation')
# Merge predictions and actual sales into one df
errors_df_hw = pd.merge(test_df, pred_fit_1, left_on='date', right_on='index')
errors_df_hw = errors_df_hw[['date', 'sales', 'pred_sales']]
errors_df_hw['errors'] = errors_df_hw.sales - errors_df_hw.pred_sales
errors_df_hw.insert(0, 'model', 'Holt-Winters')


# Evaluate the predictions for Holt-Winters without damping trend component
plt.figure(figsize=(14,7))
plt.plot(train_df['date'], train_df['sales'], label='Train')
plt.plot(test_df['date'], test_df['sales'], label='Test')
plt.plot(errors_df_hw['date'], errors_df_hw['pred_sales'], label='Forecast - HW no damping')
plt.legend(loc='best')
plt.xlabel('date')
plt.ylabel('sales')
plt.title('Forecasts using Holt-Winters without damping trend component')
plt.show()


plt.figure(figsize=(14,7))
plt.plot(errors_df_hw['date'], np.abs(errors_df_hw['errors']), label='errors')
plt.plot(errors_df_hw['date'], errors_df_hw['sales'], label='actual sales')
plt.plot(errors_df_hw['date'], errors_df_hw['pred_sales'], label='forecast')
plt.legend(loc='best')
plt.xlabel('date')
plt.ylabel('sales')
plt.title('Holt-Winters forecasts with actual sales and errors')
plt.show()

result_df_hw = errors_df_hw.groupby('model').agg(total_sales=('sales', 'sum'),
                                          total_pred_sales=('pred_sales', 'sum'),
                                          holt_winters_overall_error=('errors', 'sum'),
                                          MAE=('errors', mae),
                                          RMSE=('errors', rmse), 
                                          MAPE=('errors', mape))
result_df_hw
# Merge predictions and actual sales into one df
errors_df_hwd = pd.merge(test_df, pred_fit_2, left_on='date', right_on='index')
errors_df_hwd = errors_df_hwd[['date', 'sales','pred_sales']]
errors_df_hwd['errors'] = errors_df_hwd.sales - errors_df_hwd.pred_sales
errors_df_hwd.insert(0, 'model', 'Holt-Winters-Damped') 


# Evaluate the predictions for Holt-Winters without damping trend component
plt.figure(figsize=(14,7))
plt.plot(train_df['date'], train_df['sales'], label='Train')
plt.plot(test_df['date'], test_df['sales'], label='Test')
plt.plot(errors_df_hwd['date'], errors_df_hwd['pred_sales'], label='Forecast - HW damping')
plt.legend(loc='best')
plt.xlabel('date')
plt.ylabel('sales')
plt.title('Forecasts using Holt-Winters with damping trend component')
plt.show()

plt.figure(figsize=(14,7))
plt.plot(errors_df_hwd['date'], np.abs(errors_df_hwd['errors']), label='errors')
plt.plot(errors_df_hwd['date'], errors_df_hwd['sales'], label='actual sales')
plt.plot(errors_df_hwd['date'], errors_df_hwd['pred_sales'], label='forecast')
plt.legend(loc='best')
plt.xlabel('date')
plt.ylabel('sales')
plt.title('Holt-Winters (damping) forecasts with actual sales and errors')
plt.show()

result_df_hwd = errors_df_hwd.groupby('model').agg(total_sales=('sales', 'sum'),
                                          total_pred_sales=('pred_sales', 'sum'),
                                          holt_winters_overall_error=('errors', 'sum'),
                                          MAE=('errors', mae),
                                          RMSE=('errors', rmse), 
                                          MAPE=('errors', mape))
result_df_hwd
arima_df = train_df[['date', 'sales']].set_index('date')
arima_test_df = test_df[['date', 'sales']].set_index('date')

def test_stationarity(timeseries):
    # Plotting rolling statistics
    rollmean = timeseries.rolling(window=365).mean()
    rollstd = timeseries.rolling(window=365).std()

    plt.figure(figsize=(14,7))
    plt.plot(timeseries, color='skyblue', label='Original Series')
    plt.plot(rollmean, color='black', label='Rolling Mean')
    plt.plot(rollstd, color='red', label='Rolling Std')
    plt.legend(loc='best')
    plt.xlabel('date')
    plt.ylabel('sales')
    plt.show()
    
    # Augmented Dickey-Fuller Test
    adfuller_test = adfuller(timeseries, autolag='AIC')
    print("Test statistic = {:.3f}".format(adfuller_test[0]))
    print("P-value = {:.3f}".format(adfuller_test[1]))
    print("Critical values :")
    
    for key, value in adfuller_test[4].items():
        print("\t{}: {} - The data is {} stationary with {}% confidence"
              .format(key, value, '' if adfuller_test[0] < value else 'not', 100-int(key[:-1])))
        
    # Autocorrelation Plots
    fig, ax = plt.subplots(2, figsize=(14,7))
    ax[0] = plot_acf(timeseries, ax=ax[0], lags=20)
    ax[1] = plot_pacf(timeseries, ax=ax[1], lags=20)
    
test_stationarity(arima_df.sales)
first_difference = arima_df.sales - arima_df.sales.shift(1)
first_difference = pd.DataFrame(first_difference.dropna(inplace=False))
# Check for stationarity after differencing
test_stationarity(first_difference.sales)
arima_model61 = ARIMA(arima_df.sales, (6,1,1)).fit(disp=False)
print(arima_model61.summary())
residuals = arima_model61.resid
# Checking for seasonality
fig, ax = plt.subplots(2, figsize=(14,7))
ax[0] = plot_acf(residuals, ax=ax[0], lags=40)
ax[1] = plot_pacf(residuals, ax=ax[1], lags=40)
# fit the model
sarima_model = SARIMAX(arima_df.sales, order=(6, 1, 0), seasonal_order=(6, 1, 0, 7), 
                       enforce_invertibility=False, enforce_stationarity=False)
sarima_fit = sarima_model.fit()
arima_test_df['pred_sales'] = sarima_fit.predict(start=arima_test_df.index[0],
                                                 end=arima_test_df.index[-1], dynamic= True)
plot = sarima_fit.plot_diagnostics(figsize=(14,7))
plot
# eval
arima_test_df['errors'] = arima_test_df.sales - arima_test_df.pred_sales
arima_test_df.insert(0, 'model', 'SARIMA')

# Evaluate the predictions for Seasonal ARIMA model
plt.figure(figsize=(14,7))
plt.plot(train_df['date'], train_df['sales'], label='Train')
plt.plot(arima_test_df.index, arima_test_df['sales'], label='Test')
plt.plot(arima_test_df.index, arima_test_df['pred_sales'], label='Forecast - SARIMA')
plt.legend(loc='best')
plt.xlabel('date')
plt.ylabel('sales')
plt.title('Forecasts using Seasonal ARIMA (SARIMA) model')
plt.show()

plt.figure(figsize=(14,7))
plt.plot(arima_test_df.index, np.abs(arima_test_df['errors']), label='errors')
plt.plot(arima_test_df.index, arima_test_df['sales'], label='actual sales')
plt.plot(arima_test_df.index, arima_test_df['pred_sales'], label='forecast')
plt.legend(loc='best')
plt.xlabel('date')
plt.ylabel('sales')
plt.title('Seasonal ARIMA (SARIMA) forecasts with actual sales and errors')
plt.show()

result_df_sarima = arima_test_df.groupby('model').agg(total_sales=('sales', 'sum'),
                                          total_pred_sales=('pred_sales', 'sum'),
                                          SARIMA_overall_error=('errors', 'sum'),
                                          MAE=('errors', mae),
                                          RMSE=('errors', rmse), 
                                          MAPE=('errors', mape))
result_df_sarima
reg_df = sales_data
reg_df
# Lag features
for i in range(1,8):
    lag_i = 'lag_' + str(i)
    reg_df[lag_i] = reg_df.sales.shift(i)
    
# Rolling window
reg_df['rolling_mean'] = reg_df.sales.rolling(window=7).mean()
reg_df['rolling_max'] = reg_df.sales.rolling(window=7).max()
reg_df['rolling_min'] = reg_df.sales.rolling(window=7).min()

reg_df = reg_df.dropna(how='any', inplace=False)
reg_df = reg_df.drop(['store', 'item'], axis=1)

# Split the series to predict the last 3 months of 2017
reg_df = reg_df.set_index('date')
reg_train_df = reg_df.loc[:'2017-09-30']                        
reg_test_df = reg_df.loc['2017-10-01':]
# Correlation matrix with heatmapdata:image/png;base64,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%0AoYO8vLz073//Wx07dtSXX36pzp07q6ioSGFhYU7HFhUV6cKFCzp9+rSSk5PVtm1bx3t2u1233367%0A4/W5c+f0+OOPa9iwYbr77rsd4+fPn1ePHj0cr/39/VW3bl3H61OnTmnOnDn66quvdPXqVdlsNtWr%0AV0+SHIXR5s2bFR4erg8//FARERHy9/cv8fXHxMQoKirKaWxa32dKfDwAAADwM3sl7NS4S4UrhAID%0AA9WgQQMdO3ZMhw8f1tixYyVJb7zxho4dOyZ/f3+1bNlSvr6+js7RL9WoUUPdu3fX0qVLXc7z1Vdf%0AqX///kpMTFRsbKyjI5Sfn+/UAZLkuNXNZrNp/Pjx6tixoz744AP5+/trw4YNevXVVx37RkdHa+LE%0AiZoxY4Z27NihESNGlOr6LRaLLBZLqY4BAAAAUDoV7tY46VpX6NChQ0pOTtY999yjdu3aKTk5WYcO%0AHVJ4eLiaNGmiK1eu6MyZM45j8vLydOHCBUlSkyZNdPz4cdnt/1cBZ2VlKT8/3/G6V69eevHFF9Wl%0ASxc988wzjn0tFosyMzMd+1mtVl26dEmSlJ2drYyMDI0cOdLR5Tl27JhT9vDwcNWqVUsbNmzQN998%0Ao549e5bzpwMAAACUEM8IuVQhC6HOnTtrw4YNatasmXx9feXn56dGjRrpnXfeUXh4uO68806FhIRo%0A9uzZysnJ0aVLlzR9+nRNnjxZktSvXz/l5uZqyZIlunr1qs6cOaMxY8Zo9erVjjl8fHwkSQkJCTp+%0A/LgSExMlSd26ddPWrVv13XffKS8vT6+88oqqV68u6dptcr6+vvriiy/0008/acuWLfr666+Vl5en%0Ay5cvS5K8vb3Vv39/vfzyy+rZs6dq1qzpzo8OAAAAQAlUyEIoLCxMJ0+eVMeOHR1jHTp0UHp6usLD%0AwyVJ8+fPl91uV8+ePdW7d28VFRU5lq7+zW9+oyVLlmjXrl0KDQ3ViBEjFBkZqTFjxlw3l7+/v2bM%0AmKF58+bp5MmTGjVqlCIjI/XQQw+pT58+CgkJcdw2V6VKFSUkJGj58uXq0qWLDh06pNdee00BAQG6%0A//77HeeMjo5WXl5eqVaLAwAAAMqdzea+rZKpcM8ISdduT0tLS3Mamz59uqZPn+54HRgYqGXLlrk8%0AR+fOnfX222/f8L1fftdPr1699MUXXzhez5w5UzNnznS8fuihhxx/HjhwoAYOHOh0/AcffOD0+vvv%0A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%0AB+vjjz82OhYAAABMym5333azMjIyNG7cOIWFhSkyMlJz586VzWYr9pjz588rJCREr732WqnnoxAq%0AR3PnztVXX32ljRs36tChQxowYIDi4+OVlZVldDQAAACgQouPj1fDhg21c+dOrVy5Ujt37tTq1auL%0APWbWrFny8fEp03ymK4ROnz6tuLg4hYWFKSwsTJMmTdKlS5cc7+/Zs0c9evRQSEiInnnmGS1YsEAj%0AR44s0bk7d+6s2bNnKyAgQFWqVNGQIUP0008/6fTp07fqcgAAAACXbHYvt203Izk5WampqXryySdV%0Au3ZtNWvWTKNGjVJSUpLLYz7++GN9++236tGjR5nmNF0hNHXqVFksFu3du1fbtm3TyZMntWTJEkmS%0A1WpVfHy8Ro0apYMHD6pjx45KTEws8bl79uypli1bSpLy8vK0bNkyNWvWTG3atCnxOaxWq1JSUpw2%0AAAAAoKK70e+xVqu1RMempKQoMDBQdevWdYy1adNGJ0+eVF5e3nX7X716Vc8//7ymT5+uKlXKtv6b%0A6VaNW758uby8vFStWjX5+/urW7duOnLkiCTpwIED8vX11ciRI+Xj46MhQ4Zo48aNpZ5jzJgx+vTT%0AT9WqVSstWbJENWrUKPGxSUlJWrRokdPYIt1T6gwAAACAO1eNu9HvsY8//rji4+N/9djc3FzVqVPH%0AaeznoujChQvy8/Nzem/x4sW655571LlzZ7377rtlymu6Qujo0aOaP3++0tLSVFBQoKKiIgUHB0uS%0AsrKyFBAQ4HSfYXBwsNLS0ko1x5tvvqm8vDytWbNGI0aM0LvvvquGDRuW6NiYmBhFRUU5jf0n6vlS%0AzQ8AAAC4241+j73ttttKfLy9hCsufPvtt9qwYYO2bNlSqny/ZKpC6NKlSxo3bpxiY2O1YsUK+fn5%0A6dVXX9X+/fslSTab7brWmrd32e4e9PPz07hx47Rp0yZt3bpVcXFxJTrOYrHIYrE4jf2nTAkAAAAA%0A97nR77El5e/vr9zcXKex3NxceXl5yd/f3zFmt9uVkJCg+Pj4UhVZN2KqQkiSLl++rLi4OEd77dix%0AY4736tevr8zMTNntdnl5XWsjJicnl/i+w+joaMXHx6tnz56OMW9v7zLftwgAAADcjJtdxMBdgoOD%0Ade7cOeXk5DgKn+TkZN1xxx2qVauWY7///Oc/OnTokL755hstXLhQknTlyhV5e3vro48+0jvvvFPi%0A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%0AjhicpGBlavyLcwMfNTpGgSp9sBmAjA5BBiexrMbGrwE45vuIwUkK5h2/1egIFmmMkGWqCImIiIiI%0AiMNRRUhEREREpITSgqqWqSIkIiIiIiIORxUhEREREZESyqQxQhapIiQiIiIiIg5HFSERERERkRJK%0AY4QsU0VIREREREQcjipCIiIiIiIllMYIWaaKkIiIiIiIOBxVhERERERESiiNEbLMISpCqampNGjQ%0AgMOHDxsdRUREREREbIAqQlYUEhLCiRMn8my7evUqU6ZMoUuXLgalEhERERFHlWPSGCFL1BCyok2b%0ANuV5nJKSQkREBK1btzYokYiIiIiI3IzDNYSOHTvG+PHjSUpKAiAgIIBx48ZRqVIlALZt28a4ceM4%0Ae/Ys7du3p2bNmsTHx/Pxxx8X+VqTJ09mwIAB1KhRw6pfg4iIiIhIYageZJlDjBH6qzFjxuDm5sb2%0A7dvZuHEjR48eJTo6GoC0tDSGDx9Ov3792LlzJy1atGDp0qW3dZ0dO3awb98++vbta834IiIiIiJi%0ABQ5XEVqwYAFOTk64uLhQrVo1Wrduze7du4Hcxkv58uXp06cPpUqVolu3bqxateq2rjNv3jz69++P%0Ai4tLkZ6XlpZGenp6nm0et5VAREREREQscbiGUFJSEjNmzODAgQNcvXqV69ev07hxYwDS09OpWbMm%0ApUqVMh/fuHFjDhw4UKRrHDx4kL1795orTUURGxtLVFRUnm07m9Uq8nlERERERHLUOc4ih2oInTt3%0AjqeffprIyEjef/99XF1dmTVrFt9//z0AOTk5lC6d9yVxdi5678EvvviCVq1aUb58+SI/NyIiguDg%0A4LwbJwwr8nlERERERMQyh2oIAVy4cIGBAwfi6uoKwC+//GLeV716dU6dOoXJZMLJyQmAxMTEfI2j%0AW9m6dSvdunW7rXxubm64ubnl2ZZ5W2cSEREREUdnUkXIIoeaLCEnJwdnZ2f27NnDxYsXWbx4MRkZ%0AGWRkZHDt2jX8/Pw4ffo0y5cvJzs7m08//ZTffvutSNfIzs7m0KFDeHp63qGvQkRERERE/imHaghV%0AqVKF5557jldffZW2bdty9uxZpk+fTnZ2Nk8++SReXl5MnjyZOXPmEBAQwP79+wkLCzNXhwojMzOT%0Aa9euacpsERERETFcTjHe7I1DdI3z9PQ0T3hQr149Bg0alGf/t99+a77fuXNnunTpYm78vPzyy7i7%0Auxf6Wm5ubkWeXEFERERERIqXQ1WEbuXixYv4+/sTExNDTk4OycnJbN26laCgIKOjiYiIiIgUWQ6m%0AYrvZG4eoCBVW+fLlmT17NtOnT2fatGlUq1aNAQMG0KlTJyZOnMiKFSssPnfIkCEMHTq0GNOKiIiI%0AiMjtUkPobwIDAwkMDMy3fezYsYwdO9aARCIiIiIit0ezxlmmrnEiIiIiIuJwVBESERERESmh7HE2%0At+KiipCIiIiIiDgcVYREREREREook0ljhCxRRUhERERERByOKkIiIiIiIiWUPa7vU1xUERIRERER%0AEYejipCIiIiISAmlWeMsczJpBJWIiIiISIn0uHfnYrvWumPri+1a1qCKkB24MKm30REKVGHMJwDs%0Av6+jwUksa3hwAwAfeNr2azkw9RMe82pvdIwC/TflCwDaeLYzOIll21K3AHBhcl+DkxSswmtL7CIj%0A2PZraQ8Z4f9/v6f8n9ExClThPx8Btv1a3vh+nxv4qMFJClbpg81czThidIwClanxLwAu7/rU4CSW%0AlfV7AoBLG+cYnKRg5TqMMDqC3AY1hERERERESiiTJkuwSJMliIiIiIiIw1FFSERERESkhNL02Zap%0AIiQiIiIiIg5HFSERERERkRJKE0RbpoqQiIiIiIg4HFWERERERERKKC2oapkqQiIiIiIi4nBUERIR%0AERERKaG0jpBlqgiJiIiIiIjDUUVIRERERKSE0jpClt3RitArr7zC6NGjAZg7dy49evQAIC4ujuDg%0A4Dt5aREREREREYsMqQiFh4cTHh5uxKVFRERERByG1hGyTGOERERERETEcMePH+fpp5/moYceom3b%0AtkybNo2cnJtPAL5kyRJCQkJo3rw5kZGRJCUlFfl6RWoINWjQgMWLFxMYGMiCBQsAiI+Pp0ePHvj4%0A+BAYGMjMmTMtBr5h9erVBAQEAJCamkqDBg347rvvCA8Pp1mzZvTs2ZPU1FTz8dHR0fj5+eHv78/i%0AxYvp378/c+fOLVTm4OBgli1bRp8+fWjatCk9e/bk5MmTPP/88/j4+BASEpLnhfvhhx+IiIjAx8eH%0A1q1b8+6775r3mUwmpk+fTlBQED4+PnTp0oVdu3aZ9/fp04d58+bx4osv0rx5c1q3bs2aNWsKlVNE%0ARERExNpyMBXb7Z8aPnw47u7ubNmyhQ8//JAtW7bw0Ucf5Tvuyy+/ZO7cubz99tt8//33tG3blsGD%0A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%0AVq1a5dm+fft2pk+fzpo1awxKZl/CwsJ4//33cXNzy7N96dKlTJ8+3WbGMgGkp6fzww8/kJGRQU5O%0ADtWrV8fPz888VeylS5coV66cMpaAjGAfOZXReuwh56lTp1i/fj0pKSmULVsWb29vm1oQHXInPCop%0A42/i4+NtpoeHFANje+ZJcdi+fbvJx8fHdP78eVNsbKypUaNGps6dO5t8fX1No0ePNjpeHj/88IPp%0A1VdfNfn6+pqCgoJM06ZNM/3vf/8z/fDDD6bQ0FDTpEmTDM3XrFkz07Vr1/Jtv3btmql58+YGJLJP%0Ab731lunf//63af/+/SaTyWQ6deqUqX///qaAgADTli1bDE53a9euXTNFREQYHaNAymg99pBTGa3H%0AXnLakpI0/saWxl7JnWcbA0TkjgoMDGT9+vW4urrSo0cPqlevzv/+9z+cnZ0ZOHCg0fEAeOutt9iw%0AYQPnz5+nXbt2zJ49G39/f3M/43vvvZf333+fTp068dprrxmW09vbm6+//jpfif3bb7+lVq1aBqWy%0APy+99BL169enX79+9OzZk6VLl+Lv78+6devM6wrZgqysLN59912SkpK4evWqeXtGRobNTL+qjNZj%0ADzmV0XrsIac9rM8DJWv8jUkdpRyKGkIOYO3atYwbN47du3dz6dIl3n77bQDOnj1L9erV800daoR9%0A+/YxevRoHnvsMcqXL3/TY9zc3Bg0aFAxJ8tr+PDhjBgxgocfftg8UPXIkSN89913TJo0ydBs9qZL%0Aly7UqVOHYcOG8cgjj9jEYNq/e+ONNzh69CitW7dm4cKFPP300/zyyy9cvXrVZqZ1V0brsYecymg9%0A9pDTHtbnAViwYAEA69aty7fPycnJrhpCtrJIrRQPjRFyAB06dGDMmDEEBASwfPlyli5dSlxcHIcO%0AHeK5557j888/NzqiXdm/fz+rV68mNTWV7OxsvL29CQ0NpVmzZkZHs2mW1qs6ceIEiYmJtGvXjlKl%0ASgG2s9Bvq1at+OKLL6hSpQpNmjQhISEBgI8++ohz584xfPhwgxMqozXZQ05ltB57yNm8eXN27Nhh%0A0+vzFIU9jL9p2rQpP//8s9ExpJg43/oQsXenTp0iICAAyJ36uWPHjpQqVYoGDRpw4sQJg9PZn4YN%0AG/Lqq68SHR3NwoULef3119UIKgQXF5eb3urUqcPjjz9OuXLlzNtshclkomLFigCUKVOGixcvAtCj%0A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
corr = reg_train_df.corr()
fig = plt.figure(figsize=(10,7))
_ = sn.heatmap(corr, linewidths=.5)
X_train = reg_train_df.drop(['sales'], axis=1)
y_train = reg_train_df['sales'].values

X_test = reg_test_df.drop(['sales'], axis=1)
y_test = reg_test_df['sales'].values

#Univariate SelectKBest class to extract top 5 best features
top_features = SelectKBest(score_func=f_regression, k=5)
fit = top_features.fit(X_train, y_train)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X_train.columns)

#concat two dataframes for better visualization 
feature_scores = pd.concat([df_columns, df_scores], axis=1)
feature_scores.columns = ['Feature','Score']  # naming the dataframe columns
print(feature_scores.nlargest(5,'Score'))  # print 5 best features
# Checking for a linear relationship of the top features with sales (target variable)
fig, axs = plt.subplots(ncols=2, figsize=(14,7))
sn.scatterplot(reg_train_df.rolling_mean, reg_train_df.sales, ax=axs[0])
axs[0].set(title='Linear relationship between sales and rolling_mean of sales')
sn.scatterplot(reg_train_df.rolling_max, reg_train_df.sales, ax=axs[1])
axs[1].set(title='Linear relationship between sales and rolling_max of sales')

fig, axs = plt.subplots(ncols=2, figsize=(14,7))
sn.scatterplot(reg_train_df.rolling_min, reg_train_df.sales, ax=axs[0])
axs[0].set(title='Linear relationship between sales and rolling_min of sales')
sn.scatterplot(reg_train_df.lag_7, reg_train_df.sales, ax=axs[1])
_ = axs[1].set(title='Linear relationship between sales and lag_7 of sales')
# update X_train, X_test to include top features
X_train = X_train[['rolling_mean', 'rolling_max', 'rolling_min', 'lag_7', 'lag_1']]
X_test = X_test[['rolling_mean', 'rolling_max', 'rolling_min', 'lag_7', 'lag_1']]

# fit model
model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)

errors_df = reg_test_df[['sales']]
errors_df['pred_sales'] = preds
errors_df['errors'] = preds - y_test
errors_df.insert(0, 'model', 'LinearRegression')
# eval predictions
fig = plt.figure(figsize=(14,7))
plt.plot(reg_train_df.index, reg_train_df['sales'], label='Train')
plt.plot(reg_test_df.index, reg_test_df['sales'], label='Test')
plt.plot(errors_df.index, errors_df['pred_sales'], label='Forecast - Linear Regression')
plt.legend(loc='best')
plt.xlabel('date')
plt.ylabel('sales')
plt.title('Forecasts using Linear Regression model')
plt.show()

fig = plt.figure(figsize=(14,7))
plt.plot(errors_df.index, errors_df.errors, label='errors')
plt.plot(errors_df.index, errors_df.sales, label='actual sales')
plt.plot(errors_df.index, errors_df.pred_sales, label='forecast')
plt.legend(loc='best')
plt.xlabel('date')
plt.ylabel('sales')
plt.title('Linear Regression forecasts with actual sales and errors')
plt.show()

result_df_lr = errors_df.groupby('model').agg(total_sales=('sales', 'sum'),
                                          total_pred_sales=('pred_sales', 'sum'),
                                          LR_overall_error=('errors', 'sum'),
                                          MAE=('errors', mae),
                                          RMSE=('errors', rmse), 
                                          MAPE=('errors', mape))
result_df_lr
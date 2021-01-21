import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy.stats import normaltest
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
pd.set_option('display.max_columns', None)



def get_data(filepath):
  df = pd.read_csv(filepath)
  #print(df.head())
  return df.iloc[:,1]

df = get_data('2.csv')

def plot_timeseries(X, type='Original Data'):
  rolling_mean = X.rolling(200).mean()
  rolling_var = X.rolling(200).var()
  fig, ax = plt.subplots(figsize=(15,6))
  ax.plot(X, label='Time Series')
  ax.axhline(np.mean(X), color='r', label='Overall mean')
  ax.plot(rolling_mean, label='rolling mean')
  ax.plot(rolling_var, label='rolling var')
  ax.set_xlabel('Time')
  ax.set_ylabel('Temperature (F)')
  ax.set_title(f'Time Series ({type})')
  ax.legend()
  fig.patch.set_facecolor('w')

def plot_PACF(data):
  data = data.dropna()
  fig, ax = plt.subplots(figsize=(10,6))
  plot_pacf(data, ax=ax, lags=50)
  ax.set_xlabel('lag k')
  fig.patch.set_facecolor('w')
  plt.show()

def plot_ACF(data):
  data = data.dropna()
  fig, ax = plt.subplots(figsize=(10,6))
  plot_acf(data, ax=ax, lags=100)
  ax.set_xlabel('lag k')
  fig.patch.set_facecolor('w')
  plt.show()

df = df.to_frame(name='Temperature')
plot_timeseries(df['Temperature'], 'Original Data')

plot_ACF(df['Temperature'])

#differencing 1
df['Diff1'] = df['Temperature'] - df['Temperature'].shift(periods=1)
plot_timeseries(df['Diff1'], 'Differencing order 1')

plot_ACF(df['Diff1'])

#differencing 2
df['Diff2'] = df['Diff1'] - df['Diff1'].shift(periods=1)
plot_timeseries(df['Diff2'], 'Differencing Order 2')

plot_ACF(df['Diff2'])

#log transformation
df['Log'] = np.log(df['Temperature'])
plot_timeseries(df['Log'], 'Log transformed')

plot_ACF(df['Log'])

#log-diff1 transformation
df['Log-Diff1'] = df['Log'] - df['Log'].shift(periods=1)
plot_timeseries(df['Log-Diff1'], 'Log + Differecing order 1')

plot_ACF(df['Log-Diff1'])

# log differencing order 2
df['Log-Diff2'] = df['Log-Diff1'] - df['Log-Diff1'].shift(periods=1)
plot_timeseries(df['Log-Diff2'], 'Log + Differecing order 2')

plot_ACF(df['Log-Diff2'])

#log-differencing 24
df['Log-Diff24'] = df['Log'] - df['Log'].shift(24)
plot_timeseries(df['Log-Diff24'], 'Log + Differencing order 24')

plot_ACF(df['Log-Diff24'])

#log-differencing 24-differencing 1
df['Log-Diff24-Diff1'] = df['Log-Diff24'] - df['Log-Diff24'].shift(1)
plot_timeseries(df['Log-Diff24-Diff1'], 'Log + Differencing order 24 + Differencing order 1')

plot_ACF(df['Log-Diff24-Diff1'])

#log-diff24-diff1 is stationary. Thus, we select Log-Diff2 as our final dataset
df_final = df['Log-Diff24-Diff1'].dropna()

def get_train_test_data(data):
  df_train, df_test = train_test_split(data, train_size=0.5, shuffle=False)
  df_train = df_train.reset_index(drop=True)
  df_test = df_test.reset_index(drop=True)
  return df_train, df_test

df_train, df_test = get_train_test_data(df_final)

def plot_predicted_values(true, predicted, title, entire=True):
  if entire == False:
    true = true[:100]
    predicted = predicted[:100]
    fig = plt.figure(figsize=(10,5))
  else:
    fig = plt.figure(figsize=(20,6))    
  plt.plot(true, marker='o', label='True values')
  plt.plot(predicted, marker='x', label='Predicted values')
  plt.xlabel('Time')
  plt.ylabel('Temprature (F)')
  plt.title('True VS Predicted: ' + title)  
  plt.legend()
  fig.patch.set_facecolor('w')

"""TASK 2 (Simple moving average)"""

def simple_moving_average(data):
  rmse = []
  window =[i for i in range(1, 50)]
  for k in window:
    predicted = data.shift(1).rolling(window=k).mean()
    predicted = predicted[k:]
    true = data[k:]
    rmse.append(mean_squared_error(true, predicted, squared=False))

  fig = plt.figure(figsize=(6,4))
  plt.plot(window, rmse)
  plt.xlabel('Window size (k)')
  plt.ylabel('RMSE')
  plt.title('RMSE vs k') 
  fig.patch.set_facecolor('w')

  print(f'Min RMSE for simple moving average: {min(rmse)} at k: {window[rmse.index(min(rmse))]}')

simple_moving_average(df_train)

#lowest RMSE is observed at k=1
k = 1
true = df_train.copy()
sma_predicted = true.shift(1).rolling(window=k).mean()
sma_rmse = mean_squared_error(true[k:], sma_predicted[k:], squared=False)
print(f'Simple moving average model RMSE (train set): {sma_rmse}')

plot_predicted_values(true, sma_predicted, title='SMA', entire=True)

plot_predicted_values(true, sma_predicted, title='SMA', entire=False)

"""TASK 3 (Exponential moving average)"""

def EMA(data, alpha=0.1):
    predicted = [0] * len(data)
    predicted[0] = data[0]
    for i in range(1,len(data)):
        predicted[i] = alpha*data[i-1] + (1-alpha)*predicted[i-1]
    return predicted

def exponential_moving_average(data):
  rmse = []
  alpha = [i/10 for i in range(1,10)]
  for a in alpha:
    #predicted = data.ewm(alpha=a, adjust=False).mean()
    predicted = EMA(data, alpha=a)
    true = data
    rmse.append(mean_squared_error(true, predicted, squared=False))
  
  fig = plt.figure(figsize=(6,4))
  plt.plot(alpha, rmse)
  plt.grid()
  plt.xlabel('alpha (a)')
  plt.ylabel('RMSE')
  plt.title('RMSE vs a')  
  fig.patch.set_facecolor('w')
  print(f'Min RMSE for exponential model: {min(rmse)} at alpha: {alpha[rmse.index(min(rmse))]}')

exponential_moving_average(df_train)

a = 0.6
true = df_train.copy()
#ema_predicted = true.ewm(alpha=0.6, adjust=False).mean()
ema_predicted = EMA(true, alpha=a)
ema_rmse = mean_squared_error(true, ema_predicted, squared=False)
print(f'Exponential model RMSE (train set1): {ema_rmse}')

plot_predicted_values(true, ema_predicted, title='Exponential')

plot_predicted_values(true, ema_predicted, title='Exponential', entire=False)

"""TASK 4 (Autoregression model)"""

plot_PACF(df_train)

#first time PACF is less than 0.15 is k = 4.
def autoreg(data, p, summary=False):
  ar_model = AutoReg(data, lags=p, old_names=False).fit()
  if summary:
    print(ar_model.summary())
  return ar_model

p = 4
true = df_train.copy()
ar_model = autoreg(true, p, summary=True)
ar_predicted = ar_model.fittedvalues

ar_rmse = mean_squared_error(true[p:], ar_predicted, squared=False)
print(f'AR model RMSE (train set): {ar_rmse}')

plot_predicted_values(true, ar_predicted, title='AR')

plot_predicted_values(true, ar_predicted, title='AR', entire=False)

def plot_qq_plot(X):
  fig = sm.qqplot(X, line='45', fit=True)
  plt.title('QQ plot for the residuals')
  fig.patch.set_facecolor('w')  
  plt.show()

def plot_histogram(X):
  fig,ax = plt.subplots() 
  ax.hist(residuals, bins=10)
  ax.set_xlabel('Residual values')
  ax.set_title('Histogram for residuals')
  fig.patch.set_facecolor('w')
  plt.show()

def plot_scatter_plot(X, y_predict):
  fig, ax = plt.subplots()
  ax.scatter(y_predict, residuals)
  ax.set_ylabel('Residuals')
  ax.set_xlabel('Predicted values')
  fig.patch.set_facecolor('w')
  plt.show()

def chi_square_test(X):
  statistic, pvalue = normaltest(X)
  alpha = 0.05
  print('Chi-square test')
  print(f'pvalue: {pvalue}')
  if pvalue > alpha:
    print('Null hypothesis accepted.\nResiduals follow a normal distribution')
  elif pvalue < alpha:
    print('Null hypothesis rejected.\nResiduals do not follow a normal distribution')

residuals = ar_model.resid
plot_qq_plot(residuals)

plot_histogram(residuals)

chi_square_test(residuals)

plot_scatter_plot(residuals, ar_predicted)

"""TASK 5 (Comparison on testing models)"""

#lowest RMSE is observed at k=1
k = 1
true = df_test.copy()
sma_predicted = true.shift(1).rolling(window=k).mean()
sma_rmse = mean_squared_error(true[k:], sma_predicted[k:], squared=False)
print(f'Simple moving average RMSE (test set): {sma_rmse}')

plot_predicted_values(true,sma_predicted, title='SMA')

plot_predicted_values(true, sma_predicted, title='SMA', entire=False)

a = 0.6
true = df_test.copy()
ema_predicted = true.ewm(alpha=a, adjust=False).mean()
ema_rmse = mean_squared_error(true, ema_predicted, squared=False)
print(f'Exponential model RMSE (test set): {ema_rmse}')

plot_predicted_values(true, ema_predicted, title='Exponential')

plot_predicted_values(true, ema_predicted, title='Exponential', entire=False)

p = 4
true = df_test.copy()
ar_model = autoreg(df_test, p, summary=False)
ar_predicted = ar_model.fittedvalues
ar_rmse = mean_squared_error(true[p:], ar_predicted, squared=False)
print(f'AR model RMSE (test set): {ar_rmse}')

plot_predicted_values(true, ar_predicted, title='AR')

plot_predicted_values(true, ar_predicted, title='AR', entire=False)
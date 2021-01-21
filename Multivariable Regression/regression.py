import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import normaltest

def read_data(filepath, num_independent_variable):
  columns = []
  for i in range(1, num_independent_variable+1):
    columns.append(''.join(['X',str(i)]))
  columns.append(''.join(['Y']))
  df = pd.read_csv(filepath, header=None, names=columns)
  return df

def basic_statistics(data, **kwargs):
  columns = ['variable']
  histogram = False

  for key in kwargs.keys():
    if key!='histogram':
      columns.append(key)
    if key == 'histogram':
      histogram = True

  observations = pd.DataFrame(columns=columns)

  for i in range(1,len(data.columns)):
    v = ''.join(['X',str(i)])
    mean = np.mean(df[v])
    var = np.var(df[v])
    observations = observations.append(pd.Series([v, mean, var], index=observations.columns), ignore_index=True)

  print(observations)

  if histogram:
    fig, ax = plt.subplots(3, 2, figsize=(13,8))
    ax[0,0].hist(x=df['X1'], bins=10)
    ax[0,0].set_xlabel('Independent variable X1')
    ax[0,0].set_ylabel('Frequency')
    ax[0,0].set_title('Histogram for X1')

    ax[0,1].hist(x=df['X2'])
    ax[0,1].set_xlabel('Independent variable X2')
    ax[0,1].set_ylabel('Frequency')
    ax[0,1].set_title('Histogram for X2')


    ax[1,0].hist(x=df['X3'])
    ax[1,0].set_xlabel('Independent variable X3')
    ax[1,0].set_ylabel('Frequency')
    ax[1,0].set_title('Histogram for X3')


    ax[1,1].hist(x=df['X4'])
    ax[1,1].set_xlabel('Independent variable X4')
    ax[1,1].set_ylabel('Frequency')
    ax[1,1].set_title('Histogram for X4')


    ax[2,0].hist(x=df['X5'])
    ax[2,0].set_xlabel('Independent variable X5')
    ax[2,0].set_ylabel('Frequency')
    ax[2,0].set_title('Histogram for X5')

    fig.delaxes(ax[2,1])
    fig.tight_layout()
    fig.patch.set_facecolor('w')
    plt.show()

def plot_boxplot(data):
  bp = {}
  fig, ax = plt.subplots(3, 2, figsize=(10,7))

  bp[1] = ax[0,0].boxplot(df['X1'])
  ax[0,0].set_title('Box plot for X1')

  bp[2] = ax[0,1].boxplot(df['X2'])
  ax[0,1].set_title('Box plot for X2')

  bp[3] = ax[1,0].boxplot(df['X3'])
  ax[1,0].set_title('Box plot for X3')

  bp[4] = ax[1,1].boxplot(df['X4'])
  ax[1,1].set_title('Box plot for X4')

  bp[5] = ax[2,0].boxplot(df['X5'])
  ax[2,0].set_title('Box plot for X5')

  fig.delaxes(ax[2,1])
  fig.tight_layout()
  fig.patch.set_facecolor('w')
  plt.show()
  return bp   #returns a dict with all the box plots

def remove_outliers(boxplot, data):
  for key in boxplot.keys():
    caps = boxplot[key]['caps']
    capbottom = caps[0].get_ydata()[0]
    captop = caps[1].get_ydata()[0]

    col = ''.join(['X',str(key)])
    
    data = data[data[col] > capbottom]
    data = data[data[col] < captop]

  return data

def plot_corr_matrix(data):
  corr_matrix = data.corr()
  print(corr_matrix)

  fig, ax = plt.subplots(figsize=(8,6))
  sns.heatmap(corr_matrix, annot=True, linewidths=.5, ax=ax)
  fig.patch.set_facecolor('w')
  plt.show()

def linear_regression(X, y, constant=True):
  if constant==True:
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    model = model.fit()
    y_predict = model.predict(X)
  elif constant==False:
    model = sm.OLS(y, df[['X1', 'X2', 'X5']])
    model = model.fit()
    y_predict = model.predict(df[['X1', 'X2', 'X5']])
  
  print(model.summary())
  return (model, y_predict)

def plot_model(X, y, y_predict):
  fig, ax = plt.subplots()
  ax.scatter(X, y, marker='o')
  ax.plot(X, y_predict, 'r+', color='r')
  ax.set_xlabel('Independent variable X1')
  ax.set_ylabel('Dependent variable Y')
  ax.set_title('Regression model')
  fig.patch.set_facecolor('w')
  plt.show()

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
  ax.set_xlabel('y_predict')
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

"""Main

Task 1: Basic statistics
"""

df = read_data('2.csv', num_independent_variable=5)
df.head()

print('******************RUNNING TASK 1**************************\n\n')

basic_statistics(df, mean=True, variance=True, histogram=True)

print("BOX PLOT")
boxplot = plot_boxplot(df)
df = remove_outliers(boxplot=boxplot, data=df)

print("BOX PLOT AFTER REMOVING OUTLIERS")
boxplot = plot_boxplot(df)

plot_corr_matrix(df)

"""Task 2: simple linear regression"""

print('******************RUNNING TASK 2**************************\n\n')

X = df['X1']
y = df['Y']
(simple_model, y_predict) = linear_regression(X, y)

residuals = simple_model.resid
variance = np.var(residuals)
print(f'variance: {variance}')

plot_model(X, y, y_predict)

plot_qq_plot(residuals)

plot_histogram(residuals)

plot_scatter_plot(residuals, y_predict)

chi_square_test(residuals)

df_poly = pd.DataFrame(columns=['X1', 'X1^2'])
df_poly['X1'] = df['X1']
df_poly['X1^2'] = df['X1']**2
y = df['Y']
(polynomial_model, y_predict) = linear_regression(df_poly, y)

residuals = polynomial_model.resid
variance = np.var(residuals)
print(f'variance: {variance}')

plot_model(df['X1'], y, y_predict)

plot_qq_plot(residuals)

plot_histogram(residuals)

chi_square_test(polynomial_model.resid)

plot_scatter_plot(residuals, y_predict)

"""Task 3: multi-variable regression"""

print('******************RUNNING TASK 3**************************\n\n')

(multivariable_model,y_predict) = linear_regression(df[['X1','X2','X3','X4','X5']], df['Y'])

residuals = multivariable_model.resid
variance = np.var(residuals)
print(f'variance: {variance}')

plot_qq_plot(residuals)

plot_histogram(residuals)

plot_scatter_plot(residuals, y_predict)

chi_square_test(residuals)

multivariable_model, y_predict = linear_regression(df[['X1', 'X2', 'X5']], df['Y'])

residuals = multivariable_model.resid
variance = np.var(residuals)
print(f'variance: {variance}')

model, y_predict = linear_regression(df[['X1', 'X2', 'X5']], df['Y'], constant=False)

residuals = multivariable_model.resid
variance = np.var(residuals)
print(f'variance: {variance}')

plot_qq_plot(residuals)

plot_histogram(residuals)

chi_square_test(residuals)

plot_scatter_plot(residuals, y_predict)


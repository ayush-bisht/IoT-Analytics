import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm


filepath = '2.csv'
df = pd.read_csv(filepath, header=None, names=['X1', 'X2', 'X3', 'X4', 'X5', 'Y'])
df.head()

X = df.iloc[:,:-1]
y= df['Y']

X_train, X_test,  y_train, y_test = train_test_split(X, y, train_size=2000, random_state=10)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def grid_search(parameter_grid):
  mlp = MLPRegressor(max_iter=200, batch_size=32, random_state=10, early_stopping=True)
  clf = GridSearchCV(estimator=mlp, param_grid=parameter_grid, verbose=5, n_jobs=-1)
  clf.fit(X_train, y_train)
  # Best paramete set
  print('Best parameters found: ', clf.best_params_)
  print('Best score: ', clf.best_score_)

train_MSE, test_MSE = [], []

parameter_grid_l1 = {
  'hidden_layer_sizes': [i for i in range(5,25)],
  'alpha': 5* (10.0** -np.arange(1,6)),
  'learning_rate_init': 10.0** -np.arange(1, 6),
}

#uncomment this to perform layer 1 grid search
#grid_search(parameter_grid=parameter_grid_l1)

nn_1_hidden_layer = MLPRegressor(hidden_layer_sizes=22, alpha=5e-05, learning_rate_init=0.1,
                   max_iter=200, batch_size=32, random_state=10, early_stopping=True)

nn_1_hidden_layer.fit(X_train, y_train)

fig = plt.figure(figsize= (7,6))
x = len(nn_1_hidden_layer.loss_curve_) 
plt.plot(list(range(x-50,x)), nn_1_hidden_layer.loss_curve_[-50:])
plt.xlabel('last 50 epochs')
plt.ylabel('Loss')
plt.title('Loss curve for NN with 1 hidden layers')
fig.patch.set_facecolor('w')
plt.show()

train_MSE.append(mean_squared_error(y_train, nn_1_hidden_layer.predict(X_train)))
print(f'MSE loss for NN with 1 hidden layer: {train_MSE[0]}')

parameter_grid_l2 = {
  'hidden_layer_sizes': [ (22, i) for i in range(5,25)],
  'alpha': 5* (10.0** -np.arange(1,6)),
  'learning_rate_init': 10.0** -np.arange(1, 6),
}

#uncomment this to perfrom grid search for l2
#grid_search(parameter_grid_l2)

nn_2_hidden_layer = MLPRegressor(hidden_layer_sizes=(22,19), alpha=0.0005, learning_rate_init=0.1,
                   max_iter=200, batch_size=32, random_state=10, early_stopping=True)

nn_2_hidden_layer.fit(X_train, y_train)

fig = plt.figure(figsize= (7,6))
x = len(nn_2_hidden_layer.loss_curve_) 
plt.plot(list(range(x-50,x)), nn_2_hidden_layer.loss_curve_[-50:])
plt.xlabel('last 50 epochs')
plt.ylabel('Loss')
plt.title('Loss curve for NN with 2 hidden layers')
fig.patch.set_facecolor('w')
plt.show()

train_MSE.append(mean_squared_error(y_train, nn_2_hidden_layer.predict(X_train)))
print(f'MSE loss for NN with 2 hidden layers: {train_MSE[1]}')

parameter_grid_l3 = {
  'hidden_layer_sizes': [ (22, 19, i) for i in range(10,25)],
  'alpha': 5* (10.0** -np.arange(1,6)),
  'learning_rate_init': 10.0** -np.arange(1, 6),
}

#uncomment this to perfrom grid search for l3
#grid_search(parameter_grid_l3)

nn_3_hidden_layer = MLPRegressor(hidden_layer_sizes=(22,19,21), alpha=0.005, learning_rate_init=0.1,
                   max_iter=200, batch_size=32, random_state=10, early_stopping=True)

nn_3_hidden_layer.fit(X_train, y_train)

fig = plt.figure(figsize= (7,6))
x = len(nn_3_hidden_layer.loss_curve_) 
plt.plot(list(range(x-30,x)), nn_3_hidden_layer.loss_curve_[-30:])
plt.xlabel('last 30 epochs')
plt.ylabel('Loss')
plt.title('Loss curve for NN with 3 hidden layers')
fig.patch.set_facecolor('w')
plt.show()

train_MSE.append(mean_squared_error(y_train, nn_3_hidden_layer.predict(X_train)))
print(f'MSE loss for NN with 3 hidden layers: {train_MSE[2]}')

parameter_grid_l4 = {
  'hidden_layer_sizes': [ (22, 19, 21, i) for i in range(10,25)],
  'alpha': 5* (10.0** -np.arange(1,6)),
  'learning_rate_init': 10.0** -np.arange(1, 6),
}

#uncomment this to perfrom grid search for l2
#grid_search(parameter_grid_l4)

nn_4_hidden_layer = MLPRegressor(hidden_layer_sizes=(22,19,21,20), alpha=0.0005, learning_rate_init=0.1,
                   max_iter=200, batch_size=32, random_state=10, early_stopping=True)

nn_4_hidden_layer.fit(X_train, y_train)

fig = plt.figure(figsize= (7,6))
x = len(nn_4_hidden_layer.loss_curve_) 
plt.plot(list(range(x-20,x)), nn_4_hidden_layer.loss_curve_[-20:])
plt.xlabel('last 20 epochs')
plt.ylabel('Loss')
plt.title('Loss curve for NN with 4 hidden layers')
fig.patch.set_facecolor('w')
plt.show()

train_MSE.append(mean_squared_error(y_train, nn_4_hidden_layer.predict(X_train)))
print(f'MSE loss for NN with 4 hidden layers: {train_MSE[3]}')

y1_predict = nn_1_hidden_layer.predict(X_test)

y2_predict = nn_2_hidden_layer.predict(X_test)

y3_predict = nn_3_hidden_layer.predict(X_test)

y4_predict = nn_4_hidden_layer.predict(X_test)

test_MSE.append(mean_squared_error(y_test, y1_predict))
test_MSE.append(mean_squared_error(y_test, y2_predict))
test_MSE.append(mean_squared_error(y_test, y3_predict))
test_MSE.append(mean_squared_error(y_test, y4_predict))

print('MSE for the test set')
print(f'MSE for 1 hidden layer nn: {test_MSE[0]}')
print(f'MSE for 2 hidden layer nn: {test_MSE[1]}')
print(f'MSE for 3 hidden layer nn: {test_MSE[2]}')
print(f'MSE for 4 hidden layer nn: {test_MSE[3]}')

print(f'Best NN model has 3 hidden layers (22,19,21) with MSE loss: {test_MSE[2]}')

fig = plt.figure(figsize= (7,6))
plt.plot(np.arange(1,5), train_MSE, label='MSE for train set', marker='o')
plt.plot(np.arange(1,5), test_MSE, label='MSE for test set', marker='x')
plt.xlabel('Number of hidden layers')
plt.ylabel('Mean Squared Error')
plt.title('Train vs Test MSE comparision')
plt.legend()
fig.patch.set_facecolor('w')
plt.show()

"""Regression Model"""

def linear_regression(X, y, X_test):
  X = sm.add_constant(X)
  model = sm.OLS(y, X).fit()
  X_test = sm.add_constant(X_test)
  y_predict = model.predict(X_test)

  print(model.summary())
  return model, y_predict

#getting only X1, X2, X5
X_train_reduced = np.delete(X_train, np.s_[2:4], axis=1)
X_test_reduced = np.delete(X_test, np.s_[2:4], axis=1)

multivariable_model, y_test_predict_reg = linear_regression(X_train_reduced, y_train, X_test_reduced)

print(f'MSE loss for Regression model: {mean_squared_error(y_test, y_test_predict_reg)}')
print(f'MSE loss for the best NN mode(with 3 hidden layers: {mean_squared_error(y_test, nn_3_hidden_layer.predict(X_test))}')

print(f'SSE loss for Regression model: {sum(np.square(np.subtract(y_test, y_test_predict_reg)))}')
print(f'SSE loss for the best NN mode(with 3 hidden layers: {sum(np.square(np.subtract(y_test, nn_3_hidden_layer.predict(X_test))))}')


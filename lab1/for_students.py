import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution#####################################################

theta_best = [0, 0]

#means
x_mean = np.mean(x_train)
y_mean = np.mean(y_train)

# theta_1 (slope)
theta_1 = np.sum((x_train - x_mean) * (y_train - y_mean)) / np.sum((x_train - x_mean) ** 2)

# theta_0 (intercept)
theta_0 = y_mean - theta_1 * x_mean

theta_best = [theta_0, theta_1]

print(f"θ (closed form): θ0 -> {theta_0:.4f}, θ1 -> {theta_1:.4f}")

#########################################################################################

# TODO: calculate error#########################################################################

# mean squared error
def mean_squared_error(y_actual, y_predicted):
    return np.mean((y_actual - y_predicted) ** 2)

# predictions
y_train_pred = theta_best[0] + theta_best[1] * x_train
y_test_pred = theta_best[0] + theta_best[1] * x_test

# errors
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print(f"train MSE: {mse_train:.4f}")
print(f"test MSE: {mse_test:.4f}")

###############################################################################################

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization###################################################################################

x_mean = np.mean(x_train)
x_std = np.std(x_train)

x_train_scaled = (x_train - x_mean) / x_std
x_test_scaled = (x_test - x_mean) / x_std

print(f"standarized x_train: mean = {np.mean(x_train_scaled):.4f}, std = {np.std(x_train_scaled):.4f}")
print(f"standarized x_test: mean = {np.mean(x_test_scaled):.4f}, std = {np.std(x_test_scaled):.4f}")

############################################################################################################

# TODO: calculate theta using Batch Gradient Descent#######################################################

#parameters
learning_rate = 0.1  # step size
n_iterations = 1000  # iterations
m = len(x_train_scaled)  # training samples

# theta parameters
theta_0, theta_1 = 0, 0

# gradient descent
for iteration in range(n_iterations):
    # Compute predictions
    y_pred = theta_0 + theta_1 * x_train_scaled

    # gradients
    d_theta_0 = (1 / m) * np.sum(y_pred - y_train)
    d_theta_1 = (1 / m) * np.sum((y_pred - y_train) * x_train_scaled)

    # update theta
    theta_0 -= learning_rate * d_theta_0
    theta_1 -= learning_rate * d_theta_1

theta_best = [theta_0, theta_1]

print(f"gradient descent solution: θ0 = {theta_0:.4f}, θ1 = {theta_1:.4f}")

###############################################################################################################

# TODO: calculate error#########################################################

# MSE
def mean_squared_error(y_actual, y_predicted):
    return np.mean((y_actual - y_predicted) ** 2)

# predictions using trained θ values
y_train_pred = theta_best[0] + theta_best[1] * x_train_scaled
y_test_pred = theta_best[0] + theta_best[1] * x_test_scaled

# MSE for training and test sets
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print(f"train MSE after gradient descent: {mse_train:.4f}")
print(f"test MSE after gradient descent: {mse_test:.4f}")

#############################################################################

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
import numpy as np #mathematical functions(mean, standard deviation, vectors, matrixes)
import matplotlib.pyplot as plt #plotting

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data) #splits data into train and test data (about 80% train, 20% test)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

#LINEAR REGRESSION (closed-form solution)---------------------------------------------
#y=θ_0 +θ_1x
#y -> MPG (dependent)
#x -> Weight (independent)
#θ_0 -> intercept
#θ_1 -> slope
#θ_0 & θ_1 -> parameters to estimate
#---------------------------------------------------------------------------------------

# get the columns, conversion of data columns into numpy arrays
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: CALCULATE CLOSED-FORM SOLUTION#####################################################

theta_best = [0, 0] #initialization of model theta parameters

#means(averages)
x_mean = np.mean(x_train)
y_mean = np.mean(y_train)

# theta_1 (slope)
# θ_1 = (sum(train_x - mean_x)(train_y - mean_y)/sum(train_x - mean_x)^2)
theta_1 = np.sum((x_train - x_mean) * (y_train - y_mean)) / np.sum((x_train - x_mean) ** 2)

# theta_0 (intercept)
# θ_0 = mean_y - θ_1*mean_x
theta_0 = y_mean - theta_1 * x_mean

theta_best = [theta_0, theta_1]

print(f"θ (closed form): θ0 -> {theta_0:.4f}, θ1 -> {theta_1:.4f}")

#########################################################################################

# TODO: calculate error#########################################################################

# mean squared error
def mean_squared_error(y_actual, y_predicted): #function to compute mean squared error
    return np.mean((y_actual - y_predicted) ** 2) #squared mean

# predicted y values
# y = θ_0 + θ_1*x
#y_train_pred = theta_best[0] + theta_best[1] * x_train
y_test_pred = theta_best[0] + theta_best[1] * x_test

# mse for train and test sets using defined function
#mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

#print(f"train MSE: {mse_train:.4f}")
print(f"test MSE: {mse_test:.4f}")

###############################################################################################

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100) #generate x values for graph
y = float(theta_best[0]) + float(theta_best[1]) * x #corresponding y values using regression equation
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization###################################################################################

x_mean = np.mean(x_train) #x_train mean
x_std = np.std(x_train) #standard deviation

#standardizing by substracting the mean and dividing by standard deviaiton
x_train = (x_train - x_mean) / x_std
x_test = (x_test - x_mean) / x_std

print(f"standarized x_train: mean = {np.mean(x_train):.4f}, std = {np.std(x_train):.4f}")
print(f"standarized x_test: mean = {np.mean(x_test):.4f}, std = {np.std(x_test):.4f}")

############################################################################################################

# TODO: calculate theta using Batch Gradient Descent#######################################################

# θ_0' = θ_0 - α * 1/m * sum(y_pred - y_actual)
# θ_1' = θ_1 - α * 1/m * sum(y_pred - y_actual)*x

#parameters
learning_rate = 0.1  # step size
n_iterations = 1000  # iterations
m = len(x_train)  # training samples

# theta initialized
theta_0, theta_1 = 0, 0

# gradient descent
for iteration in range(n_iterations):
    # Compute predictions
    y_pred = theta_0 + theta_1 * x_train #prediction

    # gradients
    d_theta_0 = (1 / m) * np.sum(y_pred - y_train)
    d_theta_1 = (1 / m) * np.sum((y_pred - y_train) * x_train)

    # update theta with learning rate
    theta_0 -= learning_rate * d_theta_0
    theta_1 -= learning_rate * d_theta_1

theta_best = [theta_0, theta_1]

print(f"gradient descent solution: θ0 = {theta_0:.4f}, θ1 = {theta_1:.4f}")

###############################################################################################################

# TODO: calculate error#########################################################

#the same as before but now using gradient descent results from calculations above
# MSE
def mean_squared_error(y_actual, y_predicted): #mse
    return np.mean((y_actual - y_predicted) ** 2) #mse squared

# predictions using trained θ values
#y_train_pred = theta_best[0] + theta_best[1] * x_train_scaled
y_test_pred = theta_best[0] + theta_best[1] * x_test

# MSE for training and test sets
#mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

#print(f"train MSE after gradient descent: {mse_train:.4f}")
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
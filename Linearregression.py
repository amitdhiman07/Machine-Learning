# Building Linear regression model from scratch 
# Using Ordinary least Mean Square Approach
print("Using Ordinary least Mean Square" , end="\n")
print(' - ' * 25)
#Importing libraries
import pandas as pd 
import numpy as np
# Getting data 
dataset = pd.read_csv(r"D:\jupyter notebook\dataset\Boston_housing\housing.csv")
print(dataset)
# Let's assume were are taking RM as our X (independent variable) and MEDV as our Y (dependent variable) 
X = dataset['LSTAT'].values
print("length of x is : " , len(X))
Y = dataset['MEDV'].values
print("length of y is : " ,len(Y))
#Let's calculate the m (coefficient) and b (bias coefficient)
## y = mx + b 
# calculating the mean of X and Y 
mean_X = np.mean(X)
mean_Y = np.mean(Y)
print(f"Mean of X is : {mean_X} and Mean of Y is : {mean_Y}")
## For m (coefficient)
numerator = 0
denomenator = 0
for i in range (len(X)):
    numerator += (X[i] - mean_X) * (Y[i] - mean_Y)
    denomenator += (X[i] - mean_X) ** 2
b1 = numerator / denomenator
print(f"m Coefficient value of b1 is : {b1}" , end="\n")
b0 = mean_Y - (b1 * mean_X)
print(f"b Bias cofficient value is : {b0}")
## To Check the accuracy of our model
### Root Mean Squared Error
print("\n Root Mean Squared Error")
rmse = 0
for x in range(len(X)):
    y_pred = b1 * X[x] + b0
    rmse += (Y[x] - y_pred) ** 2
rmse = np.sqrt(rmse/len(X))
print(f"Value of RMSE is : {rmse}") 
# Calculating R2 score
sumofsquares = 0
sumofresiduals = 0
for i in range(len(X)) :
    y_pred = b0 + b1 * X[i]
    sumofsquares += (Y[i] - mean_Y) ** 2
    sumofresiduals += (Y[i] - y_pred) ** 2
score  = 1 - (sumofresiduals/sumofsquares)
print(f"\n Value of R2 score is : {score}")
    

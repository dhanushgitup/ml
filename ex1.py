# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset (Make sure the CSV file exists at the specified location)
dataset = pd.read_csv('D:\machine learning\ex1\ex1.csv')

# View the first few rows (optional)
print("Dataset preview:")
print(dataset.head())

# Separate the features (X) and target (y)
x = dataset.iloc[:, :-1].values  # Mileage
y = dataset.iloc[:, -1].values   # Selling Price

# Split the dataset into training and testing sets (2/3 training, 1/3 testing)
from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(x, y, test_size=1/3, random_state=0)

# Train the Simple Linear Regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtr, ytr)

# Make predictions on the test set
ypr = model.predict(xte)

# Print predicted and actual values
print("\nPredicted Selling Prices:")
print(ypr)
print("\nActual Selling Prices:")
print(yte)

# Visualize the training set results
plt.scatter(xtr, ytr, color='red')
plt.plot(xtr, model.predict(xtr), color='blue')
plt.title('Mileage vs Selling Price (Training set)')
plt.xlabel('Mileage of Car')
plt.ylabel('Selling Price')
plt.show()

# Visualize the test set results
plt.scatter(xte, yte, color='red')
plt.plot(xte, model.predict(xte), color='blue')
plt.title('Mileage vs Selling Price (Test set)')
plt.xlabel('Mileage of Car')
plt.ylabel('Selling Price')
plt.show()

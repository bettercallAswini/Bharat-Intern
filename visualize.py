import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

plt.style.use('bmh')

# Use a raw string to avoid issues with backslashes and read the CSV file
file_path = r'C:\Users\USER\OneDrive\Desktop\Bharat Intern\HousePrice_Prediction\train.csv'
df = pd.read_csv(file_path)

# Extract data
x = df['zip_code']
y = df['price']

# Bar chart
plt.figure(figsize=(10, 6))
plt.xlabel('Zip Code', fontsize=18)
plt.ylabel('Price ($)', fontsize=16)
plt.title('House Prices by Zip Code', fontsize=20)
plt.bar(x, y)
plt.xticks(rotation=45)
plt.show()

# Pie chart
# Note: Using a pie chart with too many categories can be visually overwhelming,
# so consider limiting the number of segments or aggregating smaller segments.

# Limit to top 10 most expensive houses for pie chart
top_10 = df.nlargest(10, 'price')
x_top_10 = top_10['zip_code']
y_top_10 = top_10['price']

plt.figure(figsize=(8, 8))
plt.title('Top 10 House Prices by Zip Code', fontsize=20)
plt.pie(y_top_10, labels=x_top_10, radius=1.2, autopct='%0.01f%%', shadow=True, explode=[0.1]*10)
plt.show()

# Line graph
plt.figure(figsize=(10, 6))
plt.xlabel('Zip Code', fontsize=18)
plt.ylabel('Price ($)', fontsize=16)
plt.title('House Prices by Zip Code', fontsize=20)
plt.plot(x, y, marker='o')
plt.xticks(rotation=45)
plt.show()

# Prepare data for prediction (using zip_code and size as features for simplicity)
X = df[['zip_code', 'size']]
y = df['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the predicted vs actual prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices', fontsize=18)
plt.ylabel('Predicted Prices', fontsize=16)
plt.title('Actual vs Predicted House Prices', fontsize=20)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.show()

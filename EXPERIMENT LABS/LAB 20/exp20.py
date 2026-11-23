# EXPERIMENT 20: FUTURE SALES PREDICTION USING LINEAR REGRESSION

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create Dataset
data = {
    'Month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
              11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'Sales': [10000, 12000, 15000, 14000, 18000,
              20000, 22000, 21000, 25000, 27000,
              30000, 28000, 32000, 35000, 33000,
              38000, 40000, 42000, 45000, 48000]
}

df = pd.DataFrame(data)

print("SALES DATASET")
print(df)

# Independent and Dependent Variables
X = df[['Month']]
y = df['Sales']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Predict Sales
y_pred = model.predict(X_test)

print("\nACTUAL SALES")
print(list(y_test))

print("\nPREDICTED SALES")
print(y_pred)

# Evaluate Model
print("\nMODEL EVALUATION")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict Future Sales
future_month = pd.DataFrame({'Month': [21]})

future_sales = model.predict(future_month)

print("\nFUTURE SALES PREDICTION")
print("Month:", 21)
print("Predicted Sales:", future_sales[0])

# Visualize Sales Prediction
plt.scatter(X, y, color='blue', label='Actual Sales')
plt.plot(X, model.predict(X), color='red', label='Predicted Sales')

plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Future Sales Prediction")
plt.legend()
plt.show()
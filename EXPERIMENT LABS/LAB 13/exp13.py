# EXPERIMENT 13: CAR PRICE PREDICTION USING LINEAR REGRESSION

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create Dataset
data = {
    'Year': [2018, 2019, 2017, 2020, 2016, 2021, 2018, 2019, 2017, 2022,
             2020, 2016, 2019, 2021, 2018, 2022, 2017, 2020, 2019, 2021],
    'Present_Price': [5.5, 7.0, 4.5, 8.0, 3.5, 10.0, 6.0, 9.0, 5.0, 12.0,
                      7.5, 4.0, 8.5, 11.0, 6.5, 13.0, 4.8, 9.5, 7.8, 10.5],
    'Kms_Driven': [25000, 30000, 40000, 20000, 50000, 15000, 35000, 22000,
                   45000, 10000, 28000, 60000, 18000, 12000, 32000, 8000,
                   42000, 16000, 26000, 14000],
    'Selling_Price': [4.5, 5.8, 3.5, 7.0, 2.8, 9.0, 4.8, 7.5, 4.0, 11.0,
                      6.5, 3.0, 7.2, 10.0, 5.5, 12.0, 3.8, 8.5, 6.8, 9.5]
}

df = pd.DataFrame(data)

print("CAR PRICE DATASET")
print(df)

# Independent and Dependent Variables
X = df[['Year', 'Present_Price', 'Kms_Driven']]
y = df['Selling_Price']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Predict Car Prices
y_pred = model.predict(X_test)

print("\nACTUAL CAR PRICES")
print(list(y_test))

print("\nPREDICTED CAR PRICES")
print(y_pred)

# Evaluate Model
print("\nMODEL EVALUATION")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict Price of a New Car
new_car = pd.DataFrame({
    'Year': [2022],
    'Present_Price': [8.5],
    'Kms_Driven': [15000]
})

predicted_price = model.predict(new_car)

print("\nNEW CAR DETAILS")
print(new_car)

print("\nPREDICTED SELLING PRICE:", predicted_price[0])
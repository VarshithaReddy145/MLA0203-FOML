# EXPERIMENT 14: HOUSE PRICE PREDICTION USING RANDOM FOREST REGRESSION

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create Dataset
data = {
    'Area': [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 1100,
             1300, 1600, 1900, 2100, 2400, 2700, 2900, 1400, 1700, 2300],
    'Bedrooms': [2, 2, 3, 3, 3, 4, 4, 4, 5, 2,
                 2, 3, 3, 4, 4, 4, 5, 3, 3, 4],
    'Bathrooms': [1, 2, 2, 2, 3, 3, 3, 4, 4, 1,
                  2, 2, 3, 3, 3, 4, 4, 2, 2, 3],
    'Price': [25, 30, 40, 45, 55, 60, 70, 80, 90, 28,
              35, 42, 50, 58, 68, 78, 88, 38, 48, 65]
}

df = pd.DataFrame(data)

print("HOUSE PRICE DATASET")
print(df)

# Independent and Dependent Variables
X = df[['Area', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train Model
model.fit(X_train, y_train)

# Predict House Prices
y_pred = model.predict(X_test)

print("\nACTUAL HOUSE PRICES")
print(list(y_test))

print("\nPREDICTED HOUSE PRICES")
print(y_pred)

# Evaluate Model
print("\nMODEL EVALUATION")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict Price of a New House
new_house = pd.DataFrame({
    'Area': [1500],
    'Bedrooms': [3],
    'Bathrooms': [2]
})

predicted_price = model.predict(new_house)

print("\nNEW HOUSE DETAILS")
print(new_house)

print("\nPREDICTED HOUSE PRICE:", predicted_price[0], "Lakhs")
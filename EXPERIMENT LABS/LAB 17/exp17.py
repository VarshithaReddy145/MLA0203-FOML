# EXPERIMENT 17: MOBILE PRICE PREDICTION USING RANDOM FOREST REGRESSION

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create Dataset
data = {
    'RAM': [2, 3, 4, 4, 6, 6, 8, 8, 8, 12,
            12, 16, 4, 6, 8, 12, 3, 4, 6, 16],
    'Storage': [32, 32, 64, 128, 64, 128, 128, 256, 128, 256,
                256, 512, 64, 128, 256, 512, 32, 128, 256, 512],
    'Battery': [3000, 3500, 4000, 4000, 4500, 5000, 4500, 5000,
                5000, 5000, 5500, 6000, 3500, 4500, 5000, 5500,
                3000, 4000, 4500, 6000],
    'Price': [8000, 10000, 12000, 14000, 16000, 18000, 22000,
              25000, 23000, 30000, 32000, 45000, 11000, 17000,
              26000, 40000, 9000, 15000, 20000, 48000]
}

df = pd.DataFrame(data)

print("MOBILE PRICE DATASET")
print(df)

# Independent and Dependent Variables
X = df[['RAM', 'Storage', 'Battery']]
y = df['Price']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Random Forest Regression Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Predict Mobile Prices
y_pred = model.predict(X_test)

print("\nACTUAL MOBILE PRICES")
print(list(y_test))

print("\nPREDICTED MOBILE PRICES")
print(y_pred)

# Evaluate Model
print("\nMODEL EVALUATION")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict Price of a New Mobile
new_mobile = pd.DataFrame({
    'RAM': [8],
    'Storage': [128],
    'Battery': [5000]
})

predicted_price = model.predict(new_mobile)

print("\nNEW MOBILE DETAILS")
print(new_mobile)

print("\nPREDICTED MOBILE PRICE: Rs.", predicted_price[0])
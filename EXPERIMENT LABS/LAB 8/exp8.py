
from sklearn.linear_model import LinearRegression

# Dataset
X = [[1], [2], [3], [4], [5]]
y = [2, 4, 6, 8, 10]

# Create the model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predict the output
prediction = model.predict([[6]])

print("Predicted value:", prediction[0])

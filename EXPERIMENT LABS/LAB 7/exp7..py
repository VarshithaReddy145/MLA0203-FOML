
from sklearn.linear_model import LogisticRegression

# Dataset
X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]

# Create the model
model = LogisticRegression()

# Train the model
model.fit(X, y)

# Predict the output
prediction = model.predict([[4]])

print("Predicted class:", prediction[0])

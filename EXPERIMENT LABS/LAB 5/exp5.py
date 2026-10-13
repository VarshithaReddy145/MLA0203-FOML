
from sklearn.neighbors import KNeighborsClassifier

# Training data
X = [[1, 2], [2, 3], [3, 1],
     [6, 5], [7, 7], [8, 6]]

# Class labels
y = ['A', 'A', 'A', 'B', 'B', 'B']

# Create KNN model
model = KNeighborsClassifier(n_neighbors=3)

# Train the model
model.fit(X, y)

# Predict a new sample
sample = [[2, 2]]

result = model.predict(sample)

print("New sample:", sample)
print("Predicted class:", result[0])

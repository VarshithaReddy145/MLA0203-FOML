
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Training data
X = [[1, 20], [2, 21], [3, 22], [6, 30],
     [7, 31], [8, 32], [2, 23], [7, 30]]

# Class labels
y = [0, 0, 0, 1, 1, 1, 0, 1]

# Create Naive Bayes model
model = GaussianNB()

# Train the model
model.fit(X, y)

# Test data
X_test = [[2, 22], [7, 31], [3, 21], [8, 32]]
y_test = [0, 1, 0, 1]

# Predict the classes
y_pred = model.predict(X_test)

# Display results
print("Actual values:", y_test)
print("Predicted values:", y_pred)

# Confusion matrix and accuracy
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

print("Accuracy:", accuracy * 100, "%")

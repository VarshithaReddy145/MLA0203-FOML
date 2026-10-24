
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Display accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy * 100, 2), "%")

# Plot actual and predicted classes
plt.figure(figsize=(8, 5))

plt.scatter(
    range(len(y_test)), y_test,
    color="blue", label="Actual Class"
)

plt.scatter(
    range(len(y_pred)), y_pred,
    color="red", marker="x", label="Predicted Class"
)

plt.xlabel("Test Sample Number")
plt.ylabel("Iris Class (0, 1, 2)")
plt.title("Experiment 12: Iris Flower Classification using KNN")
plt.yticks([0, 1, 2], iris.target_names)
plt.legend()
plt.grid(True)
plt.show()

# Predict a new flower
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)

print("Predicted Flower:", iris.target_names[prediction[0]])

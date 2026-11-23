# EXPERIMENT 18: IRIS FLOWER CLASSIFICATION USING PERCEPTRON

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Iris Dataset
iris = load_iris()

X = iris.data
y = iris.target

# Display Dataset
df = pd.DataFrame(X, columns=iris.feature_names)
df['Species'] = y

print("IRIS DATASET")
print(df.head(10))

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create Perceptron Model
model = Perceptron(max_iter=1000, random_state=42)

# Train Model
model.fit(X_train, y_train)

# Predict Values
y_pred = model.predict(X_test)

print("\nACTUAL VALUES")
print(y_test)

print("\nPREDICTED VALUES")
print(y_pred)

# Evaluate Model
print("\nMODEL EVALUATION")
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test, y_pred, target_names=iris.target_names
))

# Predict New Flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

new_flower = scaler.transform(new_flower)

result = model.predict(new_flower)

print("\nPREDICTED FLOWER:")
print(iris.target_names[result[0]])
# EXPERIMENT 15: IRIS FLOWER CLASSIFICATION USING NAIVE BAYES

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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

# Create Naive Bayes Model
model = GaussianNB()

# Train Model
model.fit(X_train, y_train)

# Predict Flower Species
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
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Predict New Flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

result = model.predict(new_flower)

print("\nPREDICTED FLOWER SPECIES:")
print(iris.target_names[result[0]])
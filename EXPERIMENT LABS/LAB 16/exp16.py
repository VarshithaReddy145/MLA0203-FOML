# EXPERIMENT 16: COMPARISON OF CLASSIFICATION ALGORITHMS

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Classification Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Compare Algorithms
results = []

print("CLASSIFICATION ALGORITHM COMPARISON")
print("-----------------------------------")

for name, model in models.items():

    # Train Model
    model.fit(X_train, y_train)

    # Predict Values
    y_pred = model.predict(X_test)

    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred) * 100

    results.append([name, accuracy])

    print("\nAlgorithm:", name)
    print("Accuracy:", accuracy, "%")

# Display Comparison Table
df = pd.DataFrame(results, columns=["Algorithm", "Accuracy"])

print("\nFINAL COMPARISON")
print(df)

# Find Best Algorithm
best_model = df.loc[df["Accuracy"].idxmax()]

print("\nBEST CLASSIFICATION ALGORITHM")
print("Algorithm:", best_model["Algorithm"])
print("Accuracy:", best_model["Accuracy"], "%")

from sklearn.tree import DecisionTreeClassifier

# Dataset: Income (in thousands), Loan Amount (in thousands)
X = [
    [20, 10],
    [25, 15],
    [30, 20],
    [50, 20],
    [60, 25],
    [70, 30],
    [22, 18],
    [65, 20]
]

# 0 = Low credit score, 1 = High credit score
y = [0, 0, 0, 1, 1, 1, 0, 1]

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Test with new customer data
income = float(input("Enter annual income (in thousands): "))
loan = float(input("Enter loan amount (in thousands): "))

prediction = model.predict([[income, loan]])

if prediction[0] == 1:
    print("Credit Score Classification: High")
else:
    print("Credit Score Classification: Low")

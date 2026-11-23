# EXPERIMENT 19: BANK LOAN PREDICTION USING NAIVE BAYES

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create Dataset
data = {
    'Income': [50000, 60000, 25000, 80000, 45000, 90000, 30000, 75000,
               55000, 40000, 85000, 35000, 65000, 95000, 28000, 70000,
               48000, 82000, 32000, 72000],
    'Credit_Score': [750, 800, 550, 780, 650, 820, 500, 760, 700, 580,
                     790, 520, 740, 850, 510, 770, 680, 810, 530, 750],
    'Employment': ['Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
                   'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes',
                   'Yes', 'Yes', 'No', 'Yes'],
    'Loan_Status': ['Approved', 'Approved', 'Rejected', 'Approved',
                    'Approved', 'Approved', 'Rejected', 'Approved',
                    'Approved', 'Rejected', 'Approved', 'Rejected',
                    'Approved', 'Approved', 'Rejected', 'Approved',
                    'Approved', 'Approved', 'Rejected', 'Approved']
}

df = pd.DataFrame(data)

print("BANK LOAN DATASET")
print(df)

# Encode Categorical Data
encoder = LabelEncoder()

df['Employment'] = encoder.fit_transform(df['Employment'])
df['Loan_Status'] = encoder.fit_transform(df['Loan_Status'])

# Independent and Dependent Variables
X = df[['Income', 'Credit_Score', 'Employment']]
y = df['Loan_Status']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Naive Bayes Model
model = GaussianNB()

# Train Model
model.fit(X_train, y_train)

# Predict Loan Status
y_pred = model.predict(X_test)

print("\nACTUAL LOAN STATUS")
print(y_test.values)

print("\nPREDICTED LOAN STATUS")
print(y_pred)

# Evaluate Model
print("\nMODEL EVALUATION")
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, zero_division=0))

# Predict New Customer Loan Status
new_customer = pd.DataFrame({
    'Income': [60000],
    'Credit_Score': [750],
    'Employment': [1]
})

result = model.predict(new_customer)

print("\nNEW CUSTOMER LOAN STATUS")

if result[0] == 0:
    print("Approved")
else:
    print("Rejected")
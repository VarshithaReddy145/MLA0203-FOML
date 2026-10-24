
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Dataset
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([2, 5, 10, 17, 26, 37])

# Linear Regression
linear = LinearRegression()
linear.fit(X, y)
linear_pred = linear.predict(X)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
poly_pred = model.predict(X_poly)

# Smooth curves for graph
X_smooth = np.linspace(1, 6, 200).reshape(-1, 1)
linear_smooth = linear.predict(X_smooth)
poly_smooth = model.predict(poly.transform(X_smooth))

# Plot graph
plt.scatter(X, y, label="Actual Data")
plt.plot(X_smooth, linear_smooth, label="Linear Regression")
plt.plot(X_smooth, poly_smooth, label="Polynomial Regression")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Experiment 9: Linear vs Polynomial Regression")
plt.legend()
plt.grid(True)
plt.show()

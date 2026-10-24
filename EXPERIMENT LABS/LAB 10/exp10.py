
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Dataset
X = np.array([
    [1, 2],
    [1.5, 1.8],
    [2, 2],
    [8, 8],
    [8.5, 8],
    [9, 8.5]
])

# Apply EM algorithm
model = GaussianMixture(
    n_components=2,
    random_state=42
)
model.fit(X)

# Predict clusters
labels = model.predict(X)

# Find cluster centers
centers = model.means_

# Plot graph
plt.scatter(
    X[:, 0], X[:, 1],
    c=labels,
    label="Data Points"
)

plt.scatter(
    centers[:, 0], centers[:, 1],
    marker="X",
    s=180,
    label="Cluster Centers"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Experiment 10: EM Algorithm Clustering")
plt.legend()
plt.grid(True)
plt.show()

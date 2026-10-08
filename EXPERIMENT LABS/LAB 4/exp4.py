import numpy as np

# -------------------------------
# 1. INPUT DATA (XOR)
# -------------------------------

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Expected output
Y = np.array([
    [0],
    [1],
    [1],
    [0]
])


# -------------------------------
# 2. SIGMOID FUNCTION
# -------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# -------------------------------
# 3. SIGMOID DERIVATIVE
# -------------------------------

def sigmoid_derivative(x):
    return x * (1 - x)


# -------------------------------
# 4. INITIALIZE WEIGHTS
# -------------------------------

np.random.seed(1)

# Input layer -> Hidden layer
W1 = np.random.rand(2, 2)

# Hidden layer -> Output layer
W2 = np.random.rand(2, 1)


# -------------------------------
# 5. INITIALIZE BIASES
# -------------------------------

b1 = np.zeros((1, 2))
b2 = np.zeros((1, 1))


# -------------------------------
# 6. LEARNING RATE
# -------------------------------

learning_rate = 0.5


# -------------------------------
# 7. TRAIN THE NETWORK
# -------------------------------

for epoch in range(10000):

    # FORWARD PROPAGATION

    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)


    # CALCULATE ERROR

    error = Y - final_output


    # BACKPROPAGATION

    output_delta = error * sigmoid_derivative(final_output)

    hidden_delta = output_delta.dot(W2.T) * \
                   sigmoid_derivative(hidden_output)


    # UPDATE WEIGHTS

    W2 = W2 + hidden_output.T.dot(output_delta) * learning_rate
    W1 = W1 + X.T.dot(hidden_delta) * learning_rate


    # UPDATE BIASES

    b2 = b2 + np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    b1 = b1 + np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate


# -------------------------------
# 8. TEST THE NETWORK
# -------------------------------

print("===================================")
print("   ARTIFICIAL NEURAL NETWORK")
print("      BACKPROPAGATION")
print("===================================")

print("\nInput\tExpected\tPredicted")
print("-----------------------------------")

for i in range(len(X)):

    prediction = final_output[i][0]

    if prediction >= 0.5:
        result = 1
    else:
        result = 0

    print(X[i], "\t", Y[i][0], "\t\t", result)

print("-----------------------------------")
print("Training completed successfully!")
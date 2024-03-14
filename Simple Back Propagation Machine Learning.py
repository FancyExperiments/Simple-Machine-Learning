import numpy as np

# Define input and output values for AND operation
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [0], [0], [1]])

# Define activation function (sigmoid)
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Initialize random weights and bias
np.random.seed(1)
weights = np.random.random((2, 1))
bias = np.random.random((1,))

# Define learning rate and number of epochs
learning_rate = 0.1
epochs = 2500

# Train the model using backpropagation algorithm
for epoch in range(epochs):
    # Feedforward
    z = np.dot(X, weights) + bias
    output = sigmoid(z)

    # Calculate error and delta
    error = output - Y
    delta = error * output * (1 - output)

    # Calculate gradient descent updates
    weight_updates = np.dot(X.T, delta)
    bias_update = np.sum(delta)

    # Update weights and bias
    weights -= learning_rate * weight_updates
    bias -= learning_rate * bias_update

    # Print progress
    if epoch % 250 == 0:
        print(f"Epoch {epoch}:  {np.mean(np.abs(error))}")

# Test the trained model
test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
prediction = sigmoid(np.dot(test_input, weights) + bias)
print("Prediction:")
print(prediction)

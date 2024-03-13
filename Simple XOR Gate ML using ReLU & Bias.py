import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Hyperparameter:
hidden_neurons = 10
epochs = 1000
learning_rate = 0.1
momentum = 0.9

# Trainingsdaten für das XOR-Gatter
training_inputs = np.array([[0,0],
                            [0,1],
                            [1,0],
                            [1,1]])

training_outputs = np.array([[0],
                             [1],
                             [1],
                             [0]])

# Gewichte und Bias initialisieren
np.random.seed(1)

weights_input_hidden = 2 * np.random.random((2, hidden_neurons)) - 1
weights_hidden_output = 2 * np.random.random((hidden_neurons, 1)) - 1

bias_hidden = np.random.random((1, hidden_neurons))
bias_output = np.random.random((1, 1))

# Momentum
weight_change_hidden = np.zeros_like(weights_input_hidden)
weight_change_output = np.zeros_like(weights_hidden_output)

for iteration in range(epochs):
    input_layer = training_inputs
    hidden_layer = relu(np.dot(input_layer, weights_input_hidden) + bias_hidden)
    output_layer = sigmoid(np.dot(hidden_layer, weights_hidden_output) + bias_output)

    # Fehler und Anpassungen berechnen
    output_error = training_outputs - output_layer
    output_adjustments = output_error * sigmoid_derivative(output_layer)

    # Fehler für die Hidden-Layer berechnen
    hidden_error = output_adjustments.dot(weights_hidden_output.T)
    hidden_adjustments = hidden_error * relu_derivative(hidden_layer)

    # Gewichte und Bias aktualisieren
    weight_change_output = (
        momentum * weight_change_output + 
        learning_rate * 
        hidden_layer.T.dot(output_adjustments)
      )

    weight_change_hidden = (
      momentum * weight_change_hidden +
      learning_rate *
      input_layer.T.dot(hidden_adjustments)
    )

    weights_hidden_output += weight_change_output

    weights_input_hidden += weight_change_hidden
    bias_hidden += learning_rate * np.sum(hidden_adjustments, axis=0, keepdims=True)

    bias_output += learning_rate * np.sum(output_adjustments, axis=0, keepdims=True)

    if iteration % 50 == 0:
        print(f"Epoche {iteration} - Fehlerrate: {np.mean(np.abs(output_error)):.3f}")

# Testen des neuronalen Netzes
for new_input, expected in zip(training_inputs, training_outputs, strict=False):
    hidden_layer = relu(np.dot(new_input, weights_input_hidden) + bias_hidden)
    output = sigmoid(np.dot(hidden_layer, weights_hidden_output) + bias_output)
    print(f"XOR {new_input} -> Ausgabe des neuronalen Netzes: ", output)
    print(f"Erwartete Ausgabe: {expected}")
    print("\r")

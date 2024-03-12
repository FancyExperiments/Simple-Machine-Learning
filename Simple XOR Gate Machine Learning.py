import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# -------------------------------------
# Hyperparameter:
# -------------------------------------
# Anzahl der Neuronen
hidden_neurons = 10

# Anzahl Epochen
epochs = 10000

# Lern-Rate
learning_rate = 0.1

# Trainingsdaten für das XOR-Gatter
training_inputs = np.array([[0,0],
                            [0,1],
                            [1,0],
                            [1,1]])

# Erwartete Ausgaben für das XOR-Gatter
training_outputs = np.array([[0],
                             [1],
                             [1],
                             [0]])

# -------------------------------------

# Gewichte initialisieren
np.random.seed(1)

# Zufällige Gewichte für die Verbindung zwischen den Hidden- und Output-Neuronen
weights_input_hidden = 2 * np.random.random((2, hidden_neurons)) - 1
weights_hidden_output = 2 * np.random.random((hidden_neurons, 1)) - 1

# Schleife für die Epochen
for iteration in range(epochs):
    input_layer = training_inputs
    hidden_layer = sigmoid(np.dot(input_layer, weights_input_hidden))
    output_layer = sigmoid(np.dot(hidden_layer, weights_hidden_output))

    # Berechnung der Fehlerwerte
    output_error = training_outputs - output_layer
    output_adjustments = output_error * sigmoid_derivative(output_layer)

    # Berechnung der Fehlerwerte für den Hidden-Layer
    hidden_error = output_adjustments.dot(weights_hidden_output.T)
    hidden_adjustments = hidden_error * sigmoid_derivative(hidden_layer)

    # Aktualisierung der Gewichte
    weights_hidden_output += learning_rate * hidden_layer.T.dot(output_adjustments)
    weights_input_hidden += learning_rate * input_layer.T.dot(hidden_adjustments)

    # Ausgabe der aktuellen Lernrate
    if iteration % 500 == 0:
        print(f"Epoche {iteration} - Fehlerrate: {np.mean(np.abs(output_error)):.3f}")

# print('Gewichte nach dem Training:')
# print(weights_input_hidden)
# print(weights_hidden_output)

# Testwerte für das XOR-Probelm
input_values = training_inputs
expected_outputs = training_outputs

# Neuronales Netzwerk testen
for new_input, expected in zip(input_values, expected_outputs):
    hidden_layer = sigmoid(np.dot(new_input, weights_input_hidden))
    output = sigmoid(np.dot(hidden_layer, weights_hidden_output))
    print(f"XOR {new_input} -> Ausgabe des neuronalen Netzes: ", round(output[0], 3))
    print(f"Erwartete Ausgabe: {expected}")
    print("\r")

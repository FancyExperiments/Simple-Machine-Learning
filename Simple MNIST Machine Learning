import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Laden des Datensatzes
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Vorverarbeitung der Trainingsdaten
train_images = train_images.reshape((-1, 28 * 28))
train_images = train_images.astype('float32') / 255

# Vorverarbeitung der Testdaten
test_images = test_images.reshape((-1, 28 * 28))
test_images = test_images.astype('float32') / 255

# Konvertierung der Labels in One-Hot-Kodierung
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# Aufbau des neuronalen Netzes
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Kompilieren des Modells
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training des Modells
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# Bewertung des Modells
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
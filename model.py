import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
print(mnist.shape())
(x_train, y_train), (x_test, y_test) = mnist.load_data()

########################################
# # Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Fixed typo here
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=18)

# Save the model
model.save('handwritten.model')
###################
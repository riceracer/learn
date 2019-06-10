"""
Intro to CNNs tutorial
From: https://www.tensorflow.org/beta/tutorials/images/intro_to_cnns
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

print(tf.__version__)


# Download MNIST dataset and reshape back into 2D images from the flat arrays
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


# Create the convolutional based model.
# This is based on intermediate layers of convolution and max pooling
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# show architecture so far
model.summary()


# Finally add Dense layers to the end of the network, which flatten the convolutional output
# and then feed the flattened vector to a final output layer matching number of classes
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='softmax'))

# print final model
model.summary()

# compile and train model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

# test the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test accuracy: {}%".format(test_acc * 100.0))

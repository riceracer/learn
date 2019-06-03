"""
Save and restore models tutorial. From:
https://www.tensorflow.org/alpha/tutorials/keras/save_and_restore_models

#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
"""

import os
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)


# Use first 1000 MNIST rows to show how to save and restore models quickly
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# create a simple model for the examples
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = create_model()
model.summary()


# You can use ModelCheckpoint callback to save the model during and at the end of training automatically
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create the callback
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                              save_weights_only=True,
                                              verbose=1)
model = create_model()
model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])

# When restoring a model from checkpoints you need to create a model with the same architecture as the original
# model. You can verify this by creating an empty model, testing accuracy, then setting the weights from
# the checkpoint and seeing the accuracy is better
model = create_model()
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# now load weights from the checkpoint
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# Checkpointing has various options you can use
# include epoch in filename (unique checkpoints)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    period=5  # save weights only every 5 epochs
)

model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_images, train_labels, epochs=50,
          callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=2)

# grab latest checkpoint
latest = tf.train.latest_checkpoint(checkpoint_dir)
latest

model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# You can also manually save weights directly without using the checkpoints callback
model.save_weights('./checkpoints/my_checkpoint')
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# You can also save the entire model as both the binary weights + the model configuration and optimizer configuration
# Save model as HDF5 standard:
model = create_model()
model.fit(train_images, train_labels, epochs=5)
model.save('my_model.h5')

# Now load the saved model from the h5 file:
new_model = keras.models.load_model('my_model.h5')
new_model.summary()

loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# Note this doesn't save the optimizer state, so if retraining the model needs to be recompiled which will
# clear the optimizer state

# Saving as a saved_model
model = create_model()
model.fit(train_images, train_labels, epochs=5)

import time
saved_model_path = "./saved_models/{}".format(int(time.time()))
tf.keras.experimental.export_saved_model(model, saved_model_path)
saved_model_path

# reload saved model
new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)
new_model.summary()

# Run the restored model
model.predict(test_images).shape

# the restored model must be compiled before evaluating
new_model.compile(optimizer=model.optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


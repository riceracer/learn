"""
TF 2.0 tutorial basic text classification. Code from:
https://www.tensorflow.org/alpha/tutorials/keras/basic_text_classification_with_tfhub

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
# Add this if GPU isn't working
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# Print debugging info
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

# Split the training set into 60% and 40%, so we'll end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_validation_split = tfds.Split.TRAIN.subsplit([6,4])

(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews",
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True)

# see some examples of the data
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
#print(train_examples_batch)
#print(train_labels_batch)

# Building the neural network - the questions are:
# 1. how to represent the text
# 2. how many layers to use in the model?
# 3. How many hidden units in each layer?

# This tutorial uses pretrained text embeddings Here are some options from TF Hub
# google/tf2-preview/gnews-swivel-20dim/1
# google/tf2-preview/gnews-swivel-20dim-with-oov/1
# google/tf2-preview/nnlm-en-dim50/1
# google/tf2-preview/nnlm-en-dim128/1

# Create a KerasLayer that takes the input text and produces and embedding:
# Regardless of input text size the output of the layer is the fixed width of the embedding
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
embedding_dim = 20
# Other potential embeddings
# embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1"
# embedding = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1"
# embedding = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
# Note - I had to add output shape to get this example to work. Must change it to match the embedding used
hub_layer = hub.KerasLayer(embedding, input_shape=[], output_shape=[embedding_dim],
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

# build the model via the layers, starting at the embedding layer
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()

# Set loss function and optimizer
# Since it is a binary model to produce a classification, use binary_crossentropy
# * mean_squared_error is another option, but binary_crossentropy is better for dealing with probabilities
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
# train for 20 epochs on batches of 512 samples
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)


# Evaluate model on test set
results = model.evaluate(test_data.batch(512), verbose=0)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))

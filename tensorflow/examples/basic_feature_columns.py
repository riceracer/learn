"""
Basic tutorial, feature columns/structured data tutorial
from: https://www.tensorflow.org/alpha/tutorials/keras/feature_columns

"""

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Load data from CSV. Copied locally to repo
# URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
URL = 'data/heart.csv'
dataframe = pd.read_csv(URL)
dataframe.head()

# split into train and test set
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


# Create input pipeline with tf.data - uses to map from columns in raw data to features
# used in the model
def df_to_dataset(dataframe, shuffle=True, batch_size=32, label_name='target'):
    dataframe = dataframe.copy()
    labels = dataframe.pop(label_name)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# look at the pipeline data
# when you call take() it returns a batch of data from the dataset
for feature_batch, label_batch in train_ds.take(1):
    print("Every feature", list(feature_batch.keys()))
    print('A batch of ages:', feature_batch['age'])
    print('A batch of targets', label_batch)


# Take one example batch for demonstration of feature columns
example_batch = next(iter(train_ds))[0]


# demo to take a feature column and transform a batch of data
def demo(feature_column, example_batch=example_batch):
    feature_layer = layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())


# numeric column
age = feature_column.numeric_column("age")
demo(age)

# bucketize column - one-hot encoding
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
demo(age_buckets)

# categorical columns one-hot encoding
thal = feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
demo(thal_one_hot)

# Use an embedding column instead of 1-hot encoding.
# Normally you should use this only when the column has a huge number of categories
# or takes a string value
# the embedding column takes the categorical column as input
thal_embedding = feature_column.embedding_column(thal, dimension=8)
demo(thal_embedding)

# Hashed features columns
# An alternative to the embedding for categorical data is use hashed feature
# columns. This can have collisions, but in practice may work well for many datasets
thal_hashed = feature_column.categorical_column_with_hash_bucket(
    'thal', hash_bucket_size=1000)
thal_hashed = feature_column.indicator_column(thal_hashed)
demo(thal_hashed)

# Crossed feature columns
# combine lists of features into single features so weights can be learned for each combination of features
# This is backed by a hash column so that the size of feature space is fixed and doesn't explode to cover
# all possible combinations
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
demo(feature_column.indicator_column(crossed_feature))

# Choose which features to use in the model
feature_columns = []
# numeric cols
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(feature_column.numeric_column(header))

# bucketized column
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# indicator columns
thal = feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# Note - not using the thal_embedding or crossed data results in better performance for this small dataset
# embedding cols
#thal_embedding = feature_column.embedding_column(thal, dimension=8)
#feature_columns.append(thal_embedding)

# crossed cols
#crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
#crossed_feature = feature_column.indicator_column(crossed_feature)
#feature_columns.append(crossed_feature)

# Create a feature layer - which converts the feature columns into a feature vector
feature_layer = layers.DenseFeatures(feature_columns)

# Now create larger batch size for actual training
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# Create compile and train the model
model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)

# print final accuracy
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)

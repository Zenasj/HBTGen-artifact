from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

mirrored_strategy = tf.distribute.MirroredStrategy()
cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()

import tensorflow as tf

# Define the distribution strategy
strategy = tf.distribute.MirroredStrategy()

# Define the model inside the strategy scope
with strategy.scope():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model as usual, but use the strategy optimizer
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Train the model using the strategy scope
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64)
with strategy.scope():
    model.fit(train_dataset, epochs=10)
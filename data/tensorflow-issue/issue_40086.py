from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf

vocabulary = ['word1', 'word2']
categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
    'feat1', vocabulary)
embedding_column = tf.feature_column.embedding_column(
    categorical_column,
    dimension=1,
    initializer=tf.constant_initializer(0))
feature_columns = [embedding_column]

# In this model, only embeddings are trainable variables
model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(feature_columns),
    tf.keras.layers.Dense(
        units=1,
        use_bias=False,
        trainable=False,
        kernel_initializer=tf.constant_initializer(1))
])

instances = {'feat1': np.array(['word1', 'word1', 'word2', 'word2'])}
labels = np.array([0, 0, 0, 0])

# Call the model to make variable initilized
model(instances)
print(model.trainable_variables)   # output weight [0, 0]

# Manually initialize mdoel weights to [-9, -7]
weights = [np.array([[-9], [-7]])]
model.layers[0].set_weights(weights)
print(model.trainable_variables)   # output weight [-9, -7]

# Fit the model using Ftrl optimizer with a small learning rate
optimizer = tf.keras.optimizers.Ftrl(learning_rate=1e-5)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
model.fit(instances, labels, epochs=1, verbose=2)
print(model.trainable_variables)  # output weight [-8.501399e-07, -7.271125e-06]
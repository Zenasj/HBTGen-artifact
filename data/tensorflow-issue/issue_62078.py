import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tf2onnx

# Create a sample RaggedTensor of lookup IDs
input_data = tf.ragged.constant([[1, 2, 3], [4, 5], [6]])

# Define the Embedding layer
embedding_dim = 4
embedding_layer = tf.keras.layers.Embedding(input_dim=7, output_dim=embedding_dim)

# Build the model
inputs = tf.keras.Input(shape=(None,), dtype=tf.int32, ragged=True)
embedded_data = embedding_layer(inputs)
model = tf.keras.Model(inputs=inputs, outputs=embedded_data)

# Export the model as SavedModel
saved_model_path = "./saved_model"
tf.saved_model.save(model, saved_model_path)

# then tensorflow-onnx could be used for converting SavedModel to onnx

import tensorflow as tf # version 2.17.1

# IDs
ids = tf.ragged.constant([[1, 2, 3], [4, 5], [6]])

# Table
weights = tf.random.uniform(shape=(16, 4), minval=0, maxval=1)

# Lookup
lookup_results = tf.nn.safe_embedding_lookup_sparse(weights, ids)
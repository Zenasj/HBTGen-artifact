import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
print(tf.__version__)

class Embedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, hidden_size):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
    
    def build(self, input_shape):
        self.shared_weights = self.add_weight(
            "weights",
            shape=(self.vocab_size, self.hidden_size),
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
                mean=0.0, 
                stddev=self.hidden_size ** (-0.5)
            )
        )
    
    def call(self, input_):
        # return tf.nn.embedding_lookup(self.shared_weights, input_)
        # return tf.gather(tf.zeros(shape=(self.vocab_size, self.hidden_size)), input_)
        return tf.gather(self.shared_weights, input_)


class SimpleModel(tf.keras.Model):
    def __init__(self, vocab_size, hidden_size):
        super(SimpleModel, self).__init__()
        self.embedding_layer = Embedding(vocab_size, hidden_size)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, ), dtype=tf.int64, name='input')])
    def call(self, input_):
        return self.embedding_layer(input_)

vocab_size = 20000
hidden_size = 300

# Building the model.
model = SimpleModel(vocab_size, hidden_size)
input_ = tf.random.uniform(shape=(20, ), dtype=tf.int64, maxval=100)
model(input_)

# Exporting to SavedModel.
saved_model_dir = 'simple_model/'
tf.saved_model.save(model, saved_model_dir)

# TFLite conversion.
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

import numpy as np
import tensorflow as tf
print(tf.__version__)

import os

class Embedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, hidden_size):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
    
    def build(self, input_shape):
        self.shared_weights = self.add_weight(
            "weights",
            shape=(self.vocab_size, self.hidden_size),
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
                mean=0.0, 
                stddev=self.hidden_size ** (-0.5)
            )
        )
    
    def call(self, input_, mode="embedding"):
        # return tf.gather(tf.identity(self.shared_weights), input_)
        # return tf.nn.embedding_lookup(self.shared_weights, input_)
        return tf.gather(self.shared_weights, input_)


class SimpleModel(tf.keras.Model):
    def __init__(self, vocab_size, hidden_size):
        super(SimpleModel, self).__init__()
        self.embedding_layer = Embedding(vocab_size, hidden_size)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 25), dtype=tf.int64, name='input')])
    def call(self, input_):
        return self.embedding_layer(input_)

vocab_size = 20000
hidden_size = 300

# Building the model.
model = SimpleModel(vocab_size, hidden_size)
input_ = tf.random.uniform(shape=(10, 25), dtype=tf.int64, maxval=100)
model(input_)

# Exporting to SavedModel.
saved_model_dir = 'simple_model/'
tf.saved_model.save(model, saved_model_dir)

# TFLite conversion.
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
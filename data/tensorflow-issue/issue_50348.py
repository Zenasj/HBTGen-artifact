# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← inferred input shape from MNIST image dimension and channel

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self, input_shape=(28, 28, 1), embedding_dim=48):
        super().__init__()
        # Feature extractor submodel (siamese branch)
        self.feature_extractor = self.build_siamese_model(input_shape, embedding_dim)
        # Output layer for similarity score
        self.output_layer = layers.Dense(1, activation="sigmoid")
    
    def build_siamese_model(self, input_shape, embedding_dim):
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Conv2D(64, (2, 2), padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.3)(x)

        pooled = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(embedding_dim)(pooled)

        model = Model(inputs, outputs)
        return model

    @staticmethod
    def euclidean_distance(vectors):
        # vectors is a list of two tensors: featsA, featsB
        featsA, featsB = vectors
        sum_squared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
        # avoid sqrt of zero by max with epsilon
        return K.sqrt(K.maximum(sum_squared, K.epsilon()))

    def call(self, inputs, training=False):
        # inputs: a list/tuple of two tensors [imgA, imgB]
        imgA, imgB = inputs
        featsA = self.feature_extractor(imgA, training=training)
        featsB = self.feature_extractor(imgB, training=training)
        # compute euclidean distance between embeddings
        distance = self.euclidean_distance([featsA, featsB])
        # apply sigmoid dense output layer to distance to get similarity score
        output = self.output_layer(distance)
        return output


def my_model_function():
    # Instantiate the model for MNIST input shape (28,28,1)
    # embedding dimension 48 as per original code
    model = MyModel(input_shape=(28,28,1), embedding_dim=48)
    return model

def GetInput():
    # Generate a random input of shape (batch, 28, 28, 1) matching MNIST format
    # Assuming batch size 4 here arbitrarily, since batch size can vary in usage
    batch_size = 4
    # Two inputs required, imgA and imgB, each is batch_size x 28 x 28 x 1
    imgA = tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)
    imgB = tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)
    return [imgA, imgB]


# tf.random.uniform((B, 86, 128, 5), dtype=tf.float32) ← Based on dummy_file.shape used in DataGenerator and model input

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Assumption: input shape is (86, 128, 5) as per DataGenerator default dim and dummy_file.shape
# This matches the example data shape and model input.

class MyModel(tf.keras.Model):
    def __init__(self, input_shape=(86, 128, 5)):
        super().__init__()
        # Build a sequential CNN as described in create_model() but encapsulated
        # to conform with subclassing and TF 2.20.0
        
        # Layers mimicking the original conv + relu + pooling stack
        self.conv_blocks = []
        
        # Conv2D(8, (2,2)), relu, maxpool 2x2 stride=1 padding=same for first layer inferred from create_model
        self.conv_blocks.append([
            layers.Conv2D(8, (2, 2), strides=(1, 1), padding="same", input_shape=input_shape),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2), strides=1)
        ])
        
        # Next conv blocks with increasing filters and max pooling
        conv_params = [
            (16, (1, 1)), 
            (32, (2, 2)),
            (64, (2, 2)),
            (128, (2, 2)),
            (256, (2, 2)),
        ]

        for filters, kernel_size in conv_params:
            self.conv_blocks.append([
                layers.Conv2D(filters, kernel_size, strides=(1, 1), padding="valid"),
                layers.ReLU(),
                layers.MaxPooling2D((2, 2), strides=1)
            ])

        # Flatten and dense layers, matching model architecture
        self.flatten = layers.Flatten()
        self.dense_layers = [
            layers.Dense(128), layers.ReLU(),
            layers.Dense(64), layers.ReLU(),
            layers.Dense(32), layers.ReLU(),
            layers.Dense(1, activation='sigmoid')
        ]

    def call(self, inputs, training=False):
        x = inputs
        # Sequentially apply conv blocks
        for block in self.conv_blocks:
            for layer in block:
                x = layer(x, training=training) if hasattr(layer, 'training') else layer(x)
        # Flatten and dense layers
        x = self.flatten(x)
        for layer in self.dense_layers:
            x = layer(x, training=training) if hasattr(layer, 'training') else layer(x)
        return x

def my_model_function():
    # Return an instance of MyModel with default shape (86,128,5)
    model = MyModel(input_shape=(86,128,5))
    # Compile with same optimizer, loss and metrics as original
    METRICS = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ]
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=METRICS)
    return model

def GetInput():
    # Generate a random input tensor with batch size 1 that matches input shape (86,128,5)
    batch_size = 1
    input_shape = (86, 128, 5)
    return tf.random.uniform((batch_size, *input_shape), dtype=tf.float32)


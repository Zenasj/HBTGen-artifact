# tf.random.uniform((batch_size, timesteps, no_of_features), dtype=tf.int32)
import tensorflow as tf
import numpy as np

# These constants are inferred from the original example:
batch_size = 32
timesteps = 3
no_of_features = 10

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Recreate original model structure: Input -> GRU(32) -> Dense(1, sigmoid)
        self.gru = tf.keras.layers.GRU(32, return_sequences=False)
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        # The single input tensor shape: (batch_size, timesteps, no_of_features)
        x = inputs
        x = tf.cast(x, tf.float32)  # Ensure float32 for GRU
        x = self.gru(x, training=training)
        out = self.dense(x)
        return out


def my_model_function():
    # Return an instance of MyModel
    
    model = MyModel()
    # To compile the model, we can add loss/optimizer similarly to the original example
    # However, since the focus is on the model structure, compilation for this code is optional
    # but we include a compile for completeness:
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[
            tf.keras.metrics.BinaryCrossentropy(name='binary_crossentropy'),
            tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
        ]
    )
    return model


def GetInput():
    # Return a random tensor that matches the expected input shape of MyModel
    
    # The original example used integer inputs with values in [0,4] – simulate similar:
    input_tensor = tf.random.uniform(
        shape=(batch_size, timesteps, no_of_features),
        minval=0,
        maxval=5,
        dtype=tf.int32,
    )
    return input_tensor


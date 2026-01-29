# tf.random.uniform((2, 3), dtype=tf.float32) for both linear and dnn inputs

import tensorflow as tf
import tensorflow_addons as tfa

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Wide (linear) part - simple linear model from keras.experimental
        self.linear_model = tf.keras.experimental.LinearModel()

        # Deep (dnn) part - simple sequential dense model
        self.dnn_model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=1)
        ])

        # Compose WideDeepModel functionality by combining outputs
        self.combined_model = tf.keras.experimental.WideDeepModel(
            self.linear_model, self.dnn_model
        )

    def call(self, inputs, training=False):
        # inputs is expected to be a list or tuple: [linear_inputs, dnn_inputs]
        # Forward pass through the WideDeepModel
        return self.combined_model(inputs, training=training)


def my_model_function():
    """
    Returns an instance of MyModel with compiled multi-optimizer wrapper
    as described in the issue discussion. The composite optimizer applies SGD
    to the first layer of dnn_model and Adam to the remaining layers.
    """
    model = MyModel()

    # Define individual optimizers
    sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

    # MultiOptimizer expects list of (optimizer, layers) tuples
    # We only apply optimizers to dnn_model layers as example from issue:
    dnn_layers = model.dnn_model.layers

    # Applying SGD to first dense layer, Adam to second dense layer
    optimizers_and_layers = [
        (sgd_optimizer, dnn_layers[0:1]),  # First Dense(64)
        (adam_optimizer, dnn_layers[1:]),  # Second Dense(1)
    ]
    multi_optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

    # Compile model with multi-optimizer, mean squared error loss and metric
    model.combined_model.compile(
        optimizer=multi_optimizer,
        loss='mse',
        metrics=['mse']
    )

    return model


def GetInput():
    """
    Returns a valid input tuple [linear_inputs, dnn_inputs] for MyModel.
    Both are float32 tensors of shape (2, 3) matching the example in the issue.
    """
    linear_inputs = tf.random.uniform(shape=(2, 3), dtype=tf.float32)
    dnn_inputs = tf.random.uniform(shape=(2, 3), dtype=tf.float32)
    return [linear_inputs, dnn_inputs]


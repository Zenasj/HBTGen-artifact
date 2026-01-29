# tf.random.uniform((100, 50), dtype=tf.float32) ← input shape inferred from reproduced example in issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Linear part of the wide and deep model
        self.linear_model = tf.keras.experimental.LinearModel()
        # Deep part of the wide and deep model: a simple Dense layer
        self.dnn_model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1)
        ])
        # WideDeepModel composes linear and deep models internally,
        # but here we explicitly manage submodules.

    def call(self, inputs, training=False):
        # Forward pass: sum outputs of linear and deep models
        # as WideDeepModel performs combined prediction.
        linear_output = self.linear_model(inputs, training=training)
        dnn_output = self.dnn_model(inputs, training=training)
        # Combine outputs additively as WideDeepModel does by default.
        combined_output = linear_output + dnn_output
        return combined_output

def my_model_function():
    # Create an instance of MyModel, compile with two optimizers
    # to reflect WideDeepModel behavior (one for linear, one for deep).
    model = MyModel()
    # Compile with two optimizers like WideDeepModel expects
    # Note: This matches WideDeepModel's multiple optimizers pattern,
    # but saving with multiple optimizers is problematic in TF as of TF 2.1.
    # This model shows how WideDeepModel composes submodels.
    model.linear_model.compile(optimizer='adagrad', loss='mse')
    model.dnn_model.compile(optimizer='rmsprop', loss='mse')
    # We compile overall model to mimic WideDeepModel compiling pattern,
    # but Keras Model doesn't officially support multiple optimizers.
    # So we store optimizers on submodels only.
    return model

def GetInput():
    # Returns a random float32 tensor matching expected input shape: (100, 50)
    return tf.random.uniform((100, 50), dtype=tf.float32)


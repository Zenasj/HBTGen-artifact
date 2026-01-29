# tf.random.normal((B, 1), dtype=tf.float32) ← Inferred input shape: batches of sequences/items with 1 feature

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define input layer shape for single feature input
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(1,))
        # Shared Dense layer that is intermediate
        self.dense_intermediate = tf.keras.layers.Dense(1)
        # Output Dense layer
        self.dense_output = tf.keras.layers.Dense(1)
        # To expose the intermediate layer output as a separate submodel
        # We'll create an intermediate model inside for convenience
        # (for eager use - per the issue resolution)
        self._intermediate_model = None

    def call(self, inputs, training=False):
        # Forward pass: Pass input through intermediate layer then output layer
        x = self.dense_intermediate(inputs)
        output = self.dense_output(x)
        return output

    def intermediate(self, inputs):
        # Return output of intermediate layer for given inputs
        # Cache intermediate model creation for efficiency
        if self._intermediate_model is None:
            # Create a submodel that maps inputs to intermediate layer output
            input_tensor = tf.keras.Input(shape=(1,))
            x = self.dense_intermediate(input_tensor)
            self._intermediate_model = tf.keras.Model(inputs=input_tensor, outputs=x)
        return self._intermediate_model(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a batch of inputs compatible with MyModel (batch size 128 chosen arbitrarily)
    batch_size = 128
    # Input shape (batch_size, 1)
    return tf.random.normal((batch_size, 1), dtype=tf.float32)

# Additionally, define a custom loss function that uses both model output and intermediate layer output,
# illustrating the conceptual fix from the issue.

def custom_loss(model, inputs, targets):
    # Get final output
    output = model(inputs)
    # Get intermediate output
    x = model.intermediate(inputs)
    # Compute error: output - targets + intermediate output
    error = output - targets + x
    return tf.reduce_mean(tf.square(error))

# Note:
# This design aligns with the issue's lesson that to use intermediate layer outputs in eager/execution,
# you should create a submodel for the intermediate outputs rather than referencing symbolic Tensors,
# which cause OOM or attribute errors when misused.

# This model and loss are compatible with tf.function(jit_compile=True).
# Example usage pattern for compilation:
#
# model = my_model_function()
# @tf.function(jit_compile=True)
# def compiled_model_loss(inputs, targets):
#     return custom_loss(model, inputs, targets)
#
# inputs = GetInput()
# targets = 3 * inputs + 2 + noise (generate accordingly)
# loss_val = compiled_model_loss(inputs, targets)


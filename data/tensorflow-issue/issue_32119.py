# tf.random.uniform((B, 1), dtype=tf.float32) ← Input shape inferred as single scalar per batch element for inputs, labels, and sample_weight

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple Dense layer to transform the input "inputs"
        self.dense = tf.keras.layers.Dense(1, use_bias=True, activation=None)

    def call(self, inputs):
        # Unpack inputs tensor tuple: (inputs, labels, sample_weights)
        # Each has shape (batch_size, 1)
        inputs_tensor, labels_tensor, sample_weights_tensor = inputs
        outputs = self.dense(inputs_tensor)
        # Custom loss weighted by sample_weights, mean over batch dimension
        loss = tf.reduce_mean(0.5 * tf.square(labels_tensor - outputs) * sample_weights_tensor)
        self.add_loss(loss)
        return outputs

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate dummy input tuple (inputs, labels, sample_weights)
    # Each is shape (batch_size=16, 1)
    batch_size = 16
    # inputs sampled as float32 random uniform scalars in range [-100, 100]
    inputs_tensor = tf.random.uniform((batch_size, 1), minval=-100.0, maxval=100.0, dtype=tf.float32)
    # labels generated as inputs tensor (identity) for demonstration
    labels_tensor = inputs_tensor
    # sample_weights all ones (equally weighted)
    sample_weights_tensor = tf.ones((batch_size, 1), dtype=tf.float32)
    return (inputs_tensor, labels_tensor, sample_weights_tensor)


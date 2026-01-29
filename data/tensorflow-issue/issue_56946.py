# tf.random.uniform((1, 1, 1, 1, 1), dtype=tf.float32) ← Input shape is a 5D tensor [batch=1,1,1,1,1]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple single Dense layer with 1 unit to mimic y = 2*x regression
        # Input shape: (1,1,1,1,1) flattened internally by Dense layer
        # Since Dense expects input (batch_size, feature_dim), we flatten the input tensor accordingly in call.
        self.dense = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        # Flatten the input tensor to 2D: (batch_size, feature_dim)
        # Here batch_size=inputs.shape[0], feature_dim=1*1*1*1=1, so flatten all but batch dimension
        batch_size = tf.shape(inputs)[0]
        x = tf.reshape(inputs, [batch_size, -1])
        # Forward through dense layer
        y = self.dense(x)
        # To keep output consistent with 5D shape, we reshape output back to (batch_size,1,1,1,1)
        y = tf.reshape(y, [batch_size, 1, 1, 1, 1])
        return y

def my_model_function():
    model = MyModel()
    # Initialize weights so the Dense layer roughly implements y=2*x
    # Dense layer weights: kernel shape == (input_dim, units)
    # Since input_dim=1, units=1, kernel is [[w]], bias is [b]
    # We want w=2, b=0
    weights = [tf.constant([[2.0]], dtype=tf.float32),
               tf.constant([0.0], dtype=tf.float32)]
    model.dense.set_weights(weights)
    return model

def GetInput():
    # Return a random tensor of shape (1,1,1,1,1) with float32 type matching model input
    # Values between -10 and 10 for demonstration
    return tf.random.uniform(shape=(1, 1, 1, 1, 1), minval=-10, maxval=10, dtype=tf.float32)


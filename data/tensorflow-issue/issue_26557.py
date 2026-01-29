# tf.random.uniform((B, 2)) ← Input shape inferred as (batch_size, 2) from the issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # First dense layer (Dense1) whose weights should be frozen (stopped gradient)
        self.dense1 = tf.keras.layers.Dense(8)
        # Lambda layer to stop gradients flowing back through Dense1's output
        self.stop_grad = tf.keras.layers.Lambda(lambda x: tf.stop_gradient(x))
        # Subsequent dense layers to be trained
        self.dense2 = tf.keras.layers.Dense(4)
        # Final dense layer with softmax activation as in original example
        self.dense3 = tf.keras.layers.Dense(1, activation='softmax')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.stop_grad(x)  # Prevent gradients w.r.t. Dense1's weights
        x = self.dense2(x)
        output = self.dense3(x)
        return output

def my_model_function():
    """
    Instantiate MyModel.
    
    Important: To emulate behavior where weights of dense1 are fixed,
    one must ensure that optimizer does not update those weights.
    Since tf.stop_gradient prevents backprop for those variables,
    tf.keras.optimizers.Adam will raise ValueError if used directly.
    
    This is the core issue described.
    To workaround:
      - use a custom training loop that excludes dense1 weights from optimizer,
      - or do not use stop_gradient and instead freeze variables via setting trainable=False.
    However, since the original issue uses stop_gradient, this model replicates that structure.
    """
    model = MyModel()
    return model

def GetInput():
    """
    Return a random input tensor of shape (batch_size, 2) matching Input(shape=(2,))
    Using batch_size = 100 as in example.
    """
    batch_size = 100
    input_tensor = tf.random.uniform((batch_size, 2), dtype=tf.float32)
    return input_tensor


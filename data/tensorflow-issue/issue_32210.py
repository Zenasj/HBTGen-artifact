# tf.random.uniform((B, 32), dtype=tf.float32) ← inferred input shape from CustomDense input_shape=(32,)

import tensorflow as tf

def projection(x):
    # Simulates a quantizer function that modifies kernels
    return 2 * x

class CustomDense(tf.keras.layers.Dense):
    def call(self, inputs):
        original_kernel = self.kernel
        # Temporarily replace self.kernel with a projected (quantized) kernel
        # Note: This pattern manipulates the layer's kernel during the forward pass
        self.kernel = projection(self.kernel)
        outputs = super().call(inputs)
        # Reset to original kernel after computation to not break eager mode
        self.kernel = original_kernel  
        return outputs

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Use the custom dense layer with output units=32, expecting input shape (32,)
        self.custom_dense = CustomDense(32)

    def call(self, inputs):
        return self.custom_dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Returns a random tensor input that matches the input expected by MyModel
    # Batch size, height, width, channels are not specified; inferred shape is (batch, 32)
    # Use batch size 4 as an example
    return tf.random.uniform((4, 32), dtype=tf.float32)


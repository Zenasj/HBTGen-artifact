# tf.random.uniform((1, 96, 96, 3), dtype=tf.float32) ← inferred input shape from issue example (Input(shape=(96,96,3)))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model as described in the issue example:
        # Input -> fake_quant_with_min_max_vars -> Conv2D(32 filters, 3x3) -> fake_quant again
        self.conv = tf.keras.layers.Conv2D(32, 3)
    
    @tf.function(jit_compile=True)
    def call(self, x):
        # Apply fake quantization with min=-3 and max=3 as in the example
        x = tf.quantization.fake_quant_with_min_max_vars(x, min=-3.0, max=3.0)
        x = self.conv(x)
        x = tf.quantization.fake_quant_with_min_max_vars(x, min=-3.0, max=3.0)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return random input with shape (1, 96, 96, 3), float32 input as model expects float inputs before quantization
    # Batch dimension assumed to be 1 as keras Input by default has batch dimension None
    return tf.random.uniform((1, 96, 96, 3), dtype=tf.float32, minval=0.0, maxval=1.0)


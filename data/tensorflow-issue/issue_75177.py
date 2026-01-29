# tf.random.uniform((1, 1), dtype=tf.float32), tf.random.uniform((1, 3), dtype=tf.float32), tf.random.uniform((1, 2), dtype=tf.float32), tf.random.uniform((1, 5), dtype=tf.float32), tf.random.uniform((1, 4), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define sigmoid activation layers for each input
        # Input names and shapes as per the issue:
        # k: (1,1), b:(1,3), m:(1,2), c:(1,5), x:(1,4)
        
        self.sigmoid_k = tf.keras.layers.Activation(tf.nn.sigmoid)
        self.sigmoid_b = tf.keras.layers.Activation(tf.nn.sigmoid)
        self.sigmoid_m = tf.keras.layers.Activation(tf.nn.sigmoid)
        self.sigmoid_c = tf.keras.layers.Activation(tf.nn.sigmoid)
        self.sigmoid_x = tf.keras.layers.Activation(tf.nn.sigmoid)
    
    @tf.function(jit_compile=True)  # XLA compilation friendly
    def call(self, inputs):
        # Inputs come as a tuple or list of 5 tensors in order:
        # (k, b, m, c, x); each tensor shape is as above.
        # Apply sigmoid activation to each input tensor.
        # Return list of tensors (like Example 1 output style).
        
        k, b, m, c, x = inputs
        out_k = self.sigmoid_k(k)  # shape (1,1)
        out_b = self.sigmoid_b(b)  # shape (1,3)
        out_m = self.sigmoid_m(m)  # shape (1,2)
        out_c = self.sigmoid_c(c)  # shape (1,5)
        out_x = self.sigmoid_x(x)  # shape (1,4)
        
        return [out_k, out_b, out_m, out_c, out_x]

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Generate a tuple of 5 random tf.float32 tensors matching the input shapes used in the issue:
    # Shapes: (1,1), (1,3), (1,2), (1,5), (1,4)
    # Use uniform random values in [0,1) matching original example
    input_k = tf.random.uniform((1,1), dtype=tf.float32)
    input_b = tf.random.uniform((1,3), dtype=tf.float32)
    input_m = tf.random.uniform((1,2), dtype=tf.float32)
    input_c = tf.random.uniform((1,5), dtype=tf.float32)
    input_x = tf.random.uniform((1,4), dtype=tf.float32)
    
    return (input_k, input_b, input_m, input_c, input_x)


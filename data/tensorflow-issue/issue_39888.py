# tf.random.uniform((32, 28, 28, 1), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple Conv2D layer as in example (filters=1, kernel_size=(1, 1))
        self.conv = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1))
        
        # Flatten and Dense layers from the second example (with 10 class outputs)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')
        
    def call(self, inputs, training=False):
        # Compute conv output - shape (batch, 28, 28, 1)
        conv_out = self.conv(inputs)
        
        # Compute flatten + dense output - shape (batch, 10)
        flat_out = self.flatten(conv_out)
        dense_out = self.dense(flat_out)
        
        # For demonstration, output both results as a tuple
        # since sample_weight behavior differs w.r.t output shape
        return conv_out, dense_out

def my_model_function():
    # Return an instance of MyModel as per above definition
    return MyModel()

def GetInput():
    # Generate input tensor matching expected shape: (32, 28, 28, 1)
    # Use float32 uniform random values
    return tf.random.uniform((32, 28, 28, 1), dtype=tf.float32)


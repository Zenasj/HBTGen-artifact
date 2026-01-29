# tf.random.uniform((B, 1), dtype=tf.float32) ← input shape inferred as (batch_size, 1) since input_shape=[1] in Dense layer

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # L2 regularizer with zero value, to replicate the reported scenario.
        # This corresponds to l2=0.0 in the original report.
        l2_regularizer = tf.keras.regularizers.L1L2(l2=0.0)
        
        # Define a simple Dense layer with 1 unit and input_shape=[1]
        # kernel_regularizer is set to l2_regularizer, which is zero.
        self.dense = tf.keras.layers.Dense(
            units=1, 
            kernel_regularizer=l2_regularizer,
            input_shape=(1,)
        )
        
    def call(self, inputs):
        # Forward pass through the Dense layer
        return self.dense(inputs)

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Generate a random input tensor matching expected input shape (batch_size, 1).
    # Batch size can be arbitrary; let's use 10 as in the example inputs np.arange(10).
    return tf.random.uniform((10, 1), dtype=tf.float32)


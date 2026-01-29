# tf.random.uniform((B, 32, 32, 1024), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        num_filters = 512
        # Conv2DTranspose layer with specified params, matching the example
        self.convTrans = tf.keras.layers.Conv2DTranspose(
            filters=num_filters,
            kernel_size=(2, 2), 
            strides=2, 
            padding="same",
            name='convTrans'
        )
    
    def call(self, inputs):
        # Forward pass through Conv2DTranspose
        return self.convTrans(inputs)

def my_model_function():
    # Instantiate the model
    # The original snippet built model via Functional API,
    # but here we implement a subclassed Model named MyModel
    return MyModel()

def GetInput():
    # Input shape expected: (B, 32, 32, 1024)
    # According to example, batch size can be 1 (or any)
    # Use tf.random.uniform with float32 dtype as training model input
    # Using batch_size=1 as reasonable default for usage/testing
    return tf.random.uniform(shape=(1, 32, 32, 1024), dtype=tf.float32)


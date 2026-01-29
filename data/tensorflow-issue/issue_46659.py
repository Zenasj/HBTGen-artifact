# tf.random.uniform((B, 81), dtype=tf.uint8)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Input layer expects shape (81,) with dtype uint8 as per provided code
        self.input_layer = tf.keras.Input(shape=(81,), dtype='uint8')
        
        # Lambda layer performing one_hot encoding with num_classes=10
        # We use a lambda to wrap keras.backend.one_hot with explicit arguments
        self.one_hot_layer = tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.one_hot(x, num_classes=10),
            output_shape=(81, 10)
        )
    
    def call(self, inputs):
        # inputs expected to be integers of dtype uint8 with shape (batch_size, 81)
        x = self.one_hot_layer(inputs)
        return x

def my_model_function():
    # Return an instance of the model
    return MyModel()

def GetInput():
    # Generate random input tensor of shape (B, 81), dtype uint8 with values [0, 9]
    # Assuming batch size B=4 for example, since batch size not specified
    batch_size = 4
    # Values 0-9 because one_hot num_classes=10
    return tf.random.uniform(
        shape=(batch_size, 81), 
        minval=0, 
        maxval=10,
        dtype=tf.int32)  # one_hot expects integer indices; uint8 converted by model input dtype


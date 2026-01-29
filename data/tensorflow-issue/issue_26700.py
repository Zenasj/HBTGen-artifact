# tf.random.uniform((B, 10), dtype=tf.float32) ← Inferred input shape based on Dense layer input_shape=(10,)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We mimic the example: a Dense layer with 11 units, use_bias=False,
        # with weights initialized to zeros as in the issue description.
        
        # Create zero weights for Dense layer: shape (input_dim=10, units=11)
        zero_weights = np.zeros((10, 11), dtype=np.float32)
        
        # Create the Dense layer with given weights and no bias.
        # Since eager mode breaks weights setting via constructor in some versions,
        # set weights explicitly after layer creation.
        self.dense = tf.keras.layers.Dense(
            units=11,
            use_bias=False,
            input_shape=(10,),
        )
        
        # Build the layer by passing a dummy input (to create weights)
        self.dense.build(input_shape=(None, 10))
        
        # Set weights to zero as specified
        self.dense.set_weights([zero_weights])
    
    @tf.function(jit_compile=True)
    def call(self, inputs):
        return self.dense(inputs)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input matching input shape (batch_size=3, features=10)
    # Using uniform random input as in the issue example (np.random.random((3, 10)))
    return tf.random.uniform((3, 10), dtype=tf.float32)


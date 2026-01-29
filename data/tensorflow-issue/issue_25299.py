# tf.random.uniform((B,), dtype=tf.float32) ← Input inferred as 1D tensor of floats (shape (batch_size,))

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A subclassed Keras Model demonstrating support for dictionary outputs and labels.
    This model takes a 1D float tensor as input and returns a dict of two outputs.
    
    This reflects the advanced use-case described in the issue discussion:
    - outputs as a dict with named keys (e.g. "output_1" and "output_2")
    - labels as a dict matching output keys to allow per-output losses
    """
    def __init__(self):
        super().__init__()
        # Simple layers for demonstration
        self.layer1 = tf.keras.layers.Dense(10, activation='relu')
        self.layer2 = tf.keras.layers.Dense(10, activation='relu')
        
    @tf.function
    def call(self, inputs, training=False):
        """
        Forward pass accepting a tensor input.
        
        Returns a dict of tensors to demonstrate multiple named outputs.
        """
        x = tf.cast(inputs, tf.float32)
        out1 = self.layer1(x)
        out2 = self.layer2(x)
        
        # Return outputs as dictionary keyed by output names
        return {"output_1": out1, "output_2": out2}

def my_model_function():
    """Factory function to return an instance of MyModel"""
    return MyModel()

def GetInput():
    """
    Returns a random tensor input compatible with MyModel.
    According to usage in examples, input is a 1D tensor (batch dimension only).
    Here we generate a batch of 5 with float32 values.
    """
    return tf.random.uniform((5,), dtype=tf.float32)


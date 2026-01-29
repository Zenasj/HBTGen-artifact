# tf.random.uniform((B, 30), dtype=tf.float32) ← input shape inferred from example data (X shape: (100, 30))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Two Dense layers with names matching output dict keys
        self.dense1 = tf.keras.layers.Dense(1, name="myname1")
        self.dense2 = tf.keras.layers.Dense(5, name="myname2")
        
    def call(self, x):
        # Return outputs as a dictionary keyed by output names to enable Keras to bind correct losses
        return {
            "myname1": self.dense1(x),
            "myname2": self.dense2(x),
        }

def my_model_function():
    # Return an instance of the subclassed model with named outputs
    return MyModel()

def GetInput():
    # Return a random input tensor matching shape (batch, 30) with float32 dtype
    # Batch size is arbitrarily chosen as 8 here for input generation
    return tf.random.uniform((8, 30), dtype=tf.float32)


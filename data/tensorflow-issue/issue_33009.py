# tf.random.uniform((1, 10), dtype=tf.float32) ← inferred input shape and dtype from issue example input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstructing the minimal architecture described in issue:
        # Input shape = (10,), Dense(32), Dense(2)
        self.dense1 = tf.keras.layers.Dense(32)
        self.dense2 = tf.keras.layers.Dense(2)
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        out = self.dense2(x)
        return out

def my_model_function():
    # Return an instance of MyModel with weights randomly initialized
    model = MyModel()
    # Build the model by running a dummy input through it once
    dummy_input = tf.zeros((1, 10), dtype=tf.float32)
    # This makes sure weights are created (build is called)
    model(dummy_input)
    return model

def GetInput():
    # Return a random tensor input matching the expected shape of MyModel (1, 10)
    # Use float32 since original code uses np.random.uniform → float32 is default TF float dtype
    return tf.random.uniform((1, 10), dtype=tf.float32)


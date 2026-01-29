# tf.random.uniform((10, 64, 64, 256), dtype=tf.float32) ← input shape inferred from Input(shape=(64, 64, 256), batch_size=10)

import tensorflow as tf
from tensorflow.keras.layers import Add

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Adding a bias as a trainable weight, shape matches input excluding batch size.
        # Note: Add weights in a way compatible with functional API; 
        #       direct addition with tf.Variable as layer input causes plotting issues.
        self.bias = self.add_weight(
            name="bias",
            shape=(10, 64, 64, 256),
            initializer="random_normal",
            trainable=True
        )
        self.add = Add()

    def call(self, x):
        # Add the bias to input tensor element-wise
        # This assumes batch size 10 fixed as per bias shape. 
        # For flexibility, broadcasting or variable batch could be considered,
        # but here we follow the example shape (fixed batch size).
        return self.add([x, self.bias])

def my_model_function():
    # Return an instance of MyModel with initialized weights
    return MyModel()

def GetInput():
    # Return a random tensor matching expected input dims: (10, 64, 64, 256) float32
    # Batch size fixed at 10 as per add_weight shape in MyModel.
    return tf.random.uniform((10, 64, 64, 256), dtype=tf.float32)


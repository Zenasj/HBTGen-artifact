# tf.random.uniform((B, input_dim), dtype=tf.float32) ← Assuming input is a 1D vector since no explicit input shape provided
import tensorflow as tf
from tensorflow.keras import layers, Model

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__(name='class_dense_model')
        # Define all Dense layers as in the original model
        self.dense_1 = layers.Dense(1)
        self.dense_64 = layers.Dense(64, activation=tf.nn.relu)
        self.dense_100 = layers.Dense(100, activation=tf.nn.relu)
        self.dense_200 = layers.Dense(200, activation=tf.nn.relu)
        self.dense_200_2 = layers.Dense(200, activation=tf.nn.relu)

    def call(self, input_tensor, training=False, **kwargs):
        # Mimic the original call logic but use the second dense_200 layer on the second call 
        # to avoid dimension mismatch as described in the issue:
        # out_4 = self.dense_200_2(out_3)
        out_1 = self.dense_64(input_tensor)
        out_2 = self.dense_100(out_1)
        out_3 = self.dense_200(out_2)
        out_4 = self.dense_200_2(out_3)
        return self.dense_1(out_4)

def my_model_function():
    # Initialize and return the model instance
    # No weights loading needed as none provided in issue
    return MyModel()

def GetInput():
    # The input shape from the original issue was not explicitly stated.
    # We infer a reasonable shape for the input as a 1D tensor of some size.
    # Since dense_64 is first, input must have dimension compatible with Dense(64).
    # Dense layer input dimension is flexible, but let's use 100 as input dimension for demonstration.
    # Batch size B is arbitrary (e.g., 2)
    B = 2
    input_dim = 100
    # Generate a random float32 tensor with shape (B, input_dim)
    return tf.random.uniform((B, input_dim), dtype=tf.float32)


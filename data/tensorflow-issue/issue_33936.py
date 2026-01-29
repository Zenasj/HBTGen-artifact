# tf.random.uniform((2, 100), dtype=tf.float32) ← Input shape inferred from batch=2, n_features=100

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Initialize the weights and bias as variables on the model
        self.n_features = 100
        # Initialize w with 1.0 for each feature and b with 1.0 as in the example
        self.w = tf.Variable(tf.ones([self.n_features], dtype=tf.float32), trainable=True)
        self.b = tf.Variable(1.0, dtype=tf.float32, trainable=True)

    def call(self, inputs, apply_sigmoid=True):
        # inputs shape: (batch_size, n_features)
        # Linear combination z = sum(w * x, axis=1) + b, output shape (batch_size, 1)
        z = tf.reduce_sum(self.w * inputs, axis=1, keepdims=True) + self.b

        if apply_sigmoid:
            # Use tf.sigmoid as in the original "error=True" case
            y_pred = tf.sigmoid(z)
        else:
            # Manual sigmoid alternative (equivalent but slower numerically)
            y_pred = 1.0 / (1.0 + tf.exp(-z))

        return y_pred

def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the input shape (batch=2, features=100), float32
    return tf.random.uniform((2, 100), dtype=tf.float32)


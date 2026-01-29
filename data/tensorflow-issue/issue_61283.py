# tf.random.uniform((1, 2), dtype=tf.float32) ← inferred input shape from issue input_shape=[1,2]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Weight shape (2, 2), bias shape (2,)
        self.w1 = tf.Variable([[3., 4.], [5., 6.]])
        self.b1 = tf.Variable([7., 8.])

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 2], dtype=tf.float32)])
    def call(self, x):
        # Simple fully connected layer compute: x*w + b
        return tf.matmul(x, self.w1) + self.b1

def my_model_function():
    # Return an instance of the defined model
    return MyModel()

def GetInput():
    # Return a tensor matching the input signature expected by MyModel:
    # shape: [1, 2], dtype: float32
    # Use tf.random.uniform to generate a sample input
    return tf.random.uniform(shape=(1, 2), dtype=tf.float32)


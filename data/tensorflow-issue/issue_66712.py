# tf.random.uniform((2, 2, 1), dtype=tf.float32), tf.random.uniform((1, 2, 2, 1), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Custom layer to perform element-wise multiplication with broadcasting
        # Equivalent to tf_input0 * tf_input1 in the original issue repro
        self.matmul = tf.keras.layers.Lambda(lambda inputs: inputs[0] * inputs[1])

    def call(self, inputs):
        tf_input0, tf_input1 = inputs
        output = self.matmul([tf_input0, tf_input1])
        return output


def my_model_function():
    # Returns an instance of MyModel
    return MyModel()


def GetInput():
    # Generate random input tensors matching the input shapes and dtypes used in the reported issue
    # input0 shape: [2, 2, 1]
    # input1 shape: [1, 2, 2, 1]
    # We assume dtype=tf.float32 as typical default for TF models and for GPU compatibility
    
    tf_input0 = tf.random.uniform(shape=(2, 2, 1), dtype=tf.float32)
    tf_input1 = tf.random.uniform(shape=(1, 2, 2, 1), dtype=tf.float32)
    
    return (tf_input0, tf_input1)


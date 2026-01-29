# tf.random.uniform((1, 4, 2), dtype=tf.float32) ← Input shape inferred from the example: batch_size=1, shape=[4,2]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable layers needed, just a lambda layer performing batched matmul with transpose of second input
        # This replicates the example from the issue demonstrating tf.linalg.matmul behavior
        self.matmul_layer = tf.keras.layers.Lambda(
            lambda x: tf.linalg.matmul(x[0][0], x[1][0], transpose_b=True)
        )
    
    def call(self, inputs):
        # inputs is expected to be a list or tuple: [input1, input2]
        # input shape: input1: (1, 4, 2), input2: (1, 4, 2)
        # We select the zeroth batch element as per the example ([0]) and do matmul with transpose
        return self.matmul_layer(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return two tensors each shaped (1, 4, 2) of float32 matching input1, input2
    # Use tf.random.uniform to generate random values for testing.
    # The example uses np.random.rand(1,4,2) cast to float32, so we follow same shape and dtype.
    input_shape = (1, 4, 2)
    input1 = tf.random.uniform(input_shape, dtype=tf.float32)
    input2 = tf.random.uniform(input_shape, dtype=tf.float32)
    return [input1, input2]


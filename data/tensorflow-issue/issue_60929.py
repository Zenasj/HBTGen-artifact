# tf.random.uniform((1, 2), dtype=tf.float32)  ← inferred from input x1 shape [1,2] and dtype float32

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Weight matrix shape: (2,2)
        self.w = tf.Variable([[3., 4.],
                              [5., 6.]])
        # Bias shape: (1,), original issue shows this shape causing TFLite failure
        # To avoid TFLite bias broadcast error, bias shape should match output dimension (2)
        # But here we keep original bias shape to replicate scenario.
        self.b = tf.Variable([3.])
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 2], dtype=tf.float32)])
    def call(self, x):
        # Perform matmul: shape (1,2) x (2,2) -> (1,2)
        # Add bias of shape (1,), which normally broadcasts in TF, but fails in TFLite due to strict check.
        return tf.matmul(x, self.w) + self.b

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor matching shape [1, 2] and dtype float32
    # Using uniform random values for general testing
    return tf.random.uniform((1, 2), dtype=tf.float32)


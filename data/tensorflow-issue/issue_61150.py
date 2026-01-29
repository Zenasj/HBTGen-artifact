# tf.random.uniform((2, 2), dtype=tf.float32) ← inferred input shape and dtype from the issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Variables as in original example: shapes are [2,1]
        self.w1 = tf.Variable([[0.], [1.]], dtype=tf.float32)
        self.m1 = tf.Variable([[1.], [1.]], dtype=tf.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=(2, 2), dtype=tf.float32)])
    def call(self, x1):
        # Add scalar tensor [1.] with broadcasting to x1: shape (2,2) + (1,)
        x2 = tf.constant([1.], shape=[1], dtype=tf.float32)  # scalar broadcast
        x3 = x1 + x2  # Broadcasting addition

        # Multiply x3 (shape [2,2]) with w1 ([2,1]) via matmul  
        # This results in shape [2,1]
        matmul_result = tf.matmul(x3, self.w1)  
        
        # Add bias m1 ([2,1])
        output = matmul_result + self.m1
        return output


def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    return model

def GetInput():
    # Return a random input tensor consistent with input shape (2, 2) and dtype float32
    return tf.random.uniform((2, 2), dtype=tf.float32)


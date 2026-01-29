# tf.random.uniform((5, 1), dtype=tf.float32)
import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable parameters—this mimics the while_loop function from the issue.
        # The model encapsulates both the loop body and condition logic.

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[5, 1], dtype=tf.float32, name='i'),
        tf.TensorSpec(shape=[5, 1], dtype=tf.float32, name='a'),
    ])
    def call(self, i, a):
        # Initialize k as zeros with shape (1,), int32
        k = tf.constant(np.zeros(1), dtype=tf.int32, shape=(1,))
        
        # Condition function: continue while k < 10
        def c(i, a, k):
            return tf.less(k, 10)

        # Body function: update i, a, and increment k by 1
        def b(i, a, k):
            return tf.add(i, a), tf.add(a, a), k + 1

        # Run the while loop with condition c and body b
        result = tf.while_loop(c, b, [i, a, k])

        # Return only the first element of the result tuple (which corresponds to updated i)
        return result[0]

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return input tensors (tuple) matching the inputs expected by MyModel.call
    # Shapes: (5,1), dtype float32
    i_input = tf.random.uniform((5, 1), dtype=tf.float32)
    a_input = tf.random.uniform((5, 1), dtype=tf.float32)
    return (i_input, a_input)


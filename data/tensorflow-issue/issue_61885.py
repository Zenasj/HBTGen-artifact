# tf.random.uniform((1, 2), dtype=tf.float32) ← Input shape inferred from the example input x1 = tf.constant([[6., 7.]], shape=[1, 2])

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Two Dense layers both with kernel and bias initialized to ones,
        # replicating the bug example setup
        self.fc1 = tf.keras.layers.Dense(
            2, 
            kernel_initializer='ones', 
            bias_initializer='ones', 
            name='fc1'
        )
        self.fc2 = tf.keras.layers.Dense(
            2, 
            kernel_initializer='ones', 
            bias_initializer='ones', 
            name='fc2'
        )
    
    @tf.function
    def call(self, x):
        # This method is decorated with @tf.function to match the "buggy" context,
        # but uses python scalars 6 and 7 instead of tf.constant([6.]) and tf.constant([7.])
        # as that was the recommended fix in the issue discussion.
        x = self.fc1(x)
        x = self.fc2(x)
        x = 6. + x  # Use python scalar addition rather than tf.constant to fix issue
        x = 7. + x
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Provide a tensor shaped [1, 2] matching the example input used in the issue.
    # Values can be arbitrary float32 numbers; using the reported example:
    # x1 = tf.constant([[6., 7.]], shape=[1, 2])
    return tf.random.uniform((1, 2), minval=0.0, maxval=10.0, dtype=tf.float32)


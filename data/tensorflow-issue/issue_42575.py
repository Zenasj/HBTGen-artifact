# tf.random.uniform((B, 10)) ← The inputs are tensors of shape (batch_size, 10)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(1)
    
    @tf.function
    def call(self, inputs, training=False):
        """
        inputs: tuple/list of two tensors a, b each of shape (batch_size, 10)
        training: boolean, controls mode behavior

        Returns:
          if training = True: tuple (dense(a+b), b)
          else: returns a tensor a

        Important notes:
        - The output structure must be consistent regardless of `training`.
        - To fix the nested structure error, always return a tuple of two elements.
          When mode is 'predict' (training=False), return (a, tf.zeros_like(b)) placeholder
          so that output shape and structure remain consistent.
        """
        a, b = inputs
        if training:
            # Return the tuple with the dense layer output and b as is
            return (self.dense(a + b), b)
        else:
            # For consistent structure, return (a, zeros_like(b)) instead of just a single tensor
            # This keeps the output nested structure identical.
            return (a, tf.zeros_like(b))

def my_model_function():
    """
    Returns an instance of MyModel.
    """
    return MyModel()

def GetInput():
    """
    Returns a tuple of two tensors (a, b), both of shape (batch_size, 10),
    suitable as input for MyModel.
    """
    # Assume batch size 2 for example
    a = tf.random.uniform((2, 10), dtype=tf.float32)
    b = tf.random.uniform((2, 10), dtype=tf.float32)
    return (a, b)


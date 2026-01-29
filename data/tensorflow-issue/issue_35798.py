# tf.random.uniform((B,), dtype=tf.int32) ← input is a 1D tensor of integers (batch dimension unknown at graph build)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

    @tf.function
    def call(self, x):
        # Use dynamic shape for batch size to avoid TensorArray init errors during model.predict
        batch_size = tf.shape(x)[0]
        result = tf.TensorArray(tf.int32, size=batch_size)
        for i in tf.range(batch_size):
            # Elementwise condition: if x[i] > 0, square it, else keep as is
            val = tf.cond(x[i] > 0,
                          true_fn=lambda: x[i] ** 2,
                          false_fn=lambda: x[i])
            result = result.write(i, val)
        return result.stack()

def my_model_function():
    # Return an instance of MyModel (no special weights or params)
    return MyModel()

def GetInput():
    # Generate a random integer vector of length 10 between -5 and 4 (like tf.range(-5,5))
    # This matches the example input and is compatible with the model.
    return tf.range(-5, 5, dtype=tf.int32)


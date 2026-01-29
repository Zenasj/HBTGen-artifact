# tf.random.uniform((10, 2), dtype=tf.float32), tf.random.uniform((1,), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No additional layers needed, simple elementwise multiply and reduction

    def call(self, inputs, training=False):
        # inputs is a list or tuple of two tensors:
        # x: shape (batch_size, feature_dim) - here (10, 2)
        # s: shape (some_shape) - here (1,), per-batch weight or parameter
        x = inputs[0]
        s = inputs[1]
        
        # Broadcast multiply x by s - s can be shape (1,) so broadcast to (batch_size, feature_dim)
        # Then reduce mean along axis=1 producing shape (batch_size,)
        return tf.reduce_mean(x * s, axis=1)

def my_model_function():
    # Return an instance of MyModel, no special initialization needed
    return MyModel()

def GetInput():
    # Return a list of two tensors matching MyModel call input signature:
    # x with shape (10, 2) representing data batch,
    # s with shape (1,) representing a per-batch parameter (broadcastable)
    x = tf.random.uniform((10, 2), dtype=tf.float32)
    s = tf.random.uniform((1,), dtype=tf.float32)
    return [x, s]


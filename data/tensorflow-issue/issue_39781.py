# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Assuming input is a 4D tensor: batch, height, width, channels

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Internal state variables
        # initialized is a boolean tf.Variable initialized to False
        self.initialized = tf.Variable(False, trainable=False, dtype=tf.bool)
        # w is a tensor variable to hold the reduced mean, initialized as None, will be created after first call
        self.w = None

    def initialize(self, x):
        # Distributed replica context
        ctx = tf.distribute.get_replica_context()
        if ctx:
            n = tf.cast(ctx.num_replicas_in_sync, tf.float32)
            # Reduce mean over axes 0,1,2 (batch, height, width) with keepdims for broadcasting
            mean = tf.reduce_mean(x, axis=[0,1,2], keepdims=True) / n
            # Perform all-reduce SUM across replicas
            # Note: ctx.all_reduce expects a single tensor, but original use is a list,
            # so unpack accordingly
            reduced_mean = ctx.all_reduce(tf.distribute.ReduceOp.SUM, mean)
            return reduced_mean
        else:
            # Non-distributed fallback
            return tf.reduce_mean(x, axis=[0,1,2], keepdims=True)

    def call(self, x, first=True):
        # On the first call (controlled by arg first), initialize internal variables
        # This approach avoids condition depending on self.initialized in the call path
        if first and not self.initialized:
            # Assign initialized to True
            self.initialized.assign(True)
            # Initialize w with the reduced mean shape and dtype matching x
            w_init = self.initialize(x)
            if self.w is None:
                self.w = tf.Variable(w_init, trainable=False, name="w_var")
            else:
                self.w.assign(w_init)
        # Subtract internal offset w from input x
        # If w not yet initialized, fallback to zeros (broadcast)
        offset = self.w if self.w is not None else tf.zeros_like(x[0:1,...])
        return x - offset

def my_model_function():
    # Create an instance of MyModel with no extra initialization needed
    return MyModel()

def GetInput():
    # Return a random float32 tensor matching (Batch, Height, Width, Channels)
    # Assume shape (4, 32, 32, 3) for example — typical small image batch size
    return tf.random.uniform((4, 32, 32, 3), dtype=tf.float32)


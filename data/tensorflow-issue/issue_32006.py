# tf.random.uniform((), dtype=tf.float32) ← Input shape is scalar float, as per the original Input(shape=())
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We encapsulate the while_loop inside a Lambda layer to avoid the "cross-device loop" error.
        # The while loop logic:
        # cond: continue while all elements of x < 10
        # body: increment i by 1, increment x by i
        self.while_layer = tf.keras.layers.Lambda(
            lambda x: tf.while_loop(
                cond=lambda i, x: tf.reduce_all(x < 10),
                body=lambda i, x: (i + 1, x + i),
                loop_vars=[tf.constant(0, dtype=tf.float32), x]
            )
        )

    def call(self, inputs):
        # Inputs is a scalar float tensor
        # Outputs a tuple (i, x) as per while_loop
        return self.while_layer(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random scalar float tensor as input matching Input(shape=())
    # Using uniform between 0 and 5 so the loop will actually run
    return tf.random.uniform(shape=(), minval=0.0, maxval=5.0, dtype=tf.float32)


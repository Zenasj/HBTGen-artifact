# tf.random.uniform((B, 100), dtype=tf.float32) ← Input shape inferred from the example with shape=(100,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Dense layer as in the original example with no bias
        self.dense1 = tf.keras.layers.Dense(100, use_bias=False)

    @tf.function
    def call(self, query):
        """
        Implements the logic from the reported issue's CustomLayer:
        - Cast input float tensor to int32
        - Expand dims axis 1 and 2 to enable broadcasting
        - Add the expanded tensors with broadcasting
        - Cast back to float32 and reduce_sum on axis 1
        - Compute rem = (query - reduced_sum) ** 2
        - Apply dense layer on rem
        """
        # Cast input float to int32 for the integer ops that cause GPU-CPU transfers
        c = tf.cast(query, tf.int32, name="castQueryToInt32")

        # Expand dims to enable broadcasting for addition
        d = tf.expand_dims(c, axis=1, name="expandDimsAxis1")
        e = tf.expand_dims(c, axis=2, name="expandDimsAxis2")

        # Broadcasted addition on integer tensors (the source of extra CPU-GPU transfer)
        g = tf.add(d, e, name="additionBroadcasted")

        # Instead of forcing extra CPU-GPU copies, cast integer tensor g to float32
        f = tf.cast(g, tf.float32, name="castGToFloat")

        # Reduce sum on axis=1
        f_reduced = tf.reduce_sum(f, axis=1, name="reduceSumF")

        # Compute squared difference with original input
        rem = (query - f_reduced) * (query - f_reduced)

        # Dense layer on the residual
        out = self.dense1(rem)

        return out

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random float32 tensor of shape (batch_size, 100)
    # Using batch_size=100 as in the original example batch_size inside model.fit
    batch_size = 100
    return tf.random.uniform((batch_size, 100), dtype=tf.float32)


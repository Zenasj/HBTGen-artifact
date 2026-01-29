# tf.random.uniform((B,), dtype=dt) where B is arbitrary batch size, dt in {tf.float32, tf.int64, tf.int32}

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # We define three Scale layers with different dtypes to reflect the issue:
        # float32 and int64 behave fine with jit_compile=True, int32 fails on GPU.
        # This model encapsulates all three scales to show/test this behavior.
        self.scale_float32 = Scale(2, tf.float32)
        self.scale_int64 = Scale(2, tf.int64)
        self.scale_int32 = Scale(2, tf.int32)

    @tf.function(jit_compile=True)
    def call(self, x):
        # We produce the output of all three Scale layers
        # Note: The int32 branch will raise the known runtime error in GPU+jit_compile.
        out_float32 = self.scale_float32(x)
        out_int64 = self.scale_int64(tf.cast(x, tf.int64))
        out_int32 = self.scale_int32(tf.cast(x, tf.int32))
        # Return all three results in a tuple so user can observe behavior/compare
        return out_float32, out_int64, out_int32


class Scale(tf.keras.layers.Layer):
    def __init__(self, c, dt):
        super(Scale, self).__init__(name='Scale')
        # Create a scalar weight with initial constant c of dtype dt
        self.c = self.add_weight(
            initializer=tf.keras.initializers.Constant(c), dtype=dt, shape=())
        
    @tf.function(jit_compile=True)
    def call(self, x):
        return self.c * x


def my_model_function():
    # Return an instance of MyModel that includes all 3 dtype scale layers
    return MyModel()

def GetInput():
    # Return a batch input vector of ones with dtype float32, shape (4,)
    # Must be compatible with MyModel: will be cast internally for int64 and int32 branches
    return tf.ones(4, dtype=tf.float32)


import tensorflow as tf

class Model(tf.Module):
    @tf.function
    def __call__(self, x, rlim):
        return tf.raw_ops.Conv2D(
            # `rlim` > index_max means `min(index_max, rlim)` which is still valid in slicing.
            input=x[:, :, :rlim, :],
            filter=tf.zeros([1, 1, 1, 1]),
            strides=[1, 2, 2, 1],
            padding="SAME",
        )

[1,2,3][:2**31] == [1,2,3] # True
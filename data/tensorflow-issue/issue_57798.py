import tensorflow as tf
print(tf.__version__)

class MyModule(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function
    def __call__(self, x, y, z):
        return tf.raw_ops.UnsortedSegmentProd(
            data=x, segment_ids=y, num_segments=z,
        )


inp = {
    "x": tf.constant([3]),
    "y": tf.constant([1], dtype=tf.int64),
    "z": tf.constant(0x7fffffff + 1, dtype=tf.int64),
}
m = MyModule()

out = m(**inp)  # Error!
print(out)
print(out.shape)
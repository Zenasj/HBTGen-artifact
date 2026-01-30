import tensorflow as tf


class A(tf.experimental.ExtensionType):
    a1: tf.Tensor

    class Spec:
        def __init__(self):
            self.a1 = tf.TensorSpec(shape=[2, 11], dtype=tf.float16)


class B(tf.experimental.ExtensionType):
    b1: tf.Tensor
    b2: tf.Tensor
    b3: tf.Tensor
    b4: tf.Tensor
    b5: tf.Tensor
    b6: tf.Tensor
    b7: tf.Tensor
    b8: tf.Tensor
    b9: tf.Tensor
    b10: tf.Tensor
    a: A

    class Spec:
        def __init__(self):
            self.b1 = tf.TensorSpec(shape=[2, 1], dtype=tf.float32)
            self.b2 = tf.TensorSpec(shape=[2, 2], dtype=tf.float16)
            self.b3 = tf.TensorSpec(shape=[2, 3], dtype=tf.int8)
            self.b4 = tf.TensorSpec(shape=[2, 4], dtype=tf.int16)
            self.b5 = tf.TensorSpec(shape=[2, 5], dtype=tf.int32)
            self.b6 = tf.TensorSpec(shape=[2, 6], dtype=tf.bfloat16)
            self.b7 = tf.TensorSpec(shape=[2, 7], dtype=tf.float64)
            self.b8 = tf.TensorSpec(shape=[2, 8], dtype=tf.int64)
            self.b9 = tf.TensorSpec(shape=[2, 9], dtype=tf.uint8)
            self.b10 = tf.TensorSpec(shape=[2, 10], dtype=tf.uint16)
            self.a = A.Spec()


def loading_fn():
    objects = B(
        1 * tf.ones(shape=[2, 1], dtype=tf.float32),
        2 * tf.ones(shape=[2, 2], dtype=tf.float16),
        3 * tf.ones(shape=[2, 3], dtype=tf.int8),
        4 * tf.ones(shape=[2, 4], dtype=tf.int16),
        5 * tf.ones(shape=[2, 5], dtype=tf.int32),
        6 * tf.ones(shape=[2, 6], dtype=tf.bfloat16),
        7 * tf.ones(shape=[2, 7], dtype=tf.float64),
        8 * tf.ones(shape=[2, 8], dtype=tf.int64),
        9 * tf.ones(shape=[2, 9], dtype=tf.uint8),
        10 * tf.ones(shape=[2, 10], dtype=tf.uint16),
        A(11 * tf.ones(shape=[2, 11], dtype=tf.float16)),
    )

    return objects


def func(i):
    return tf.py_function(loading_fn, inp=[], Tout=B.Spec())


dataset = tf.data.Dataset.from_tensor_slices([i for i in range(10)])
dataset = dataset.map(func, num_parallel_calls=10)

for _ in range(10):
    for path in iter(dataset):
        pass
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def preprocess(x, l):
    return tf.image.convert_image_dtype(x, tf.float32), l

train_data, test_data = tf.keras.datasets.mnist.load_data()
train_data = tf.data.Dataset.from_tensor_slices(train_data).map(preprocess)

@tf.function
def run_loop(model, data):
    res = True
    for x, y in data:
        batch_size = int(x.shape[0])
        res = model(x)[0] == 1.
    return res

model = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='softmax')])
result = run_loop(model, train_data.batch(32, drop_remainder=False)) # drop_remainder=True for "EagerTensor" exception

"""Returns a `tf.TensorShape` that represents the shape of this tensor.

    >>> t = tf.constant([1,2,3,4,5])
    >>> t.shape
    TensorShape([5])

    `tf.Tensor.shape` is equivalent to `tf.Tensor.get_shape()`.

    In a `tf.function` or when building a model using
    `tf.keras.Input`, they return the build-time shape of the
    tensor, which may be partially unknown.

    A `tf.TensorShape` is not a tensor. Use `tf.shape(t)` to get a tensor
    containing the shape, calculated at runtime.

    See `tf.Tensor.get_shape()`, and `tf.TensorShape` for details and examples.
    """
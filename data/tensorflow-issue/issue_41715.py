import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import tensorflow_datasets as tfds

policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
tf.keras.mixed_precision.experimental.set_policy(policy)


def preprocessing(data):
    return tf.cast(data["image"], tf.float32) / 255.0, data["label"]


dataset = (
    tfds.load("cifar10", split="train")
    .map(preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .cache()
    .batch(1024)
    .prefetch(1)
)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu", input_shape=(32, 32, 3)),
        tf.keras.layers.DepthwiseConv2D(3, depth_multiplier=8, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Activation("softmax", dtype="float32"),
    ]
)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

model.fit(dataset, epochs=3, callbacks=[tf.keras.callbacks.TensorBoard("logs")])

import tensorflow as tf
import time

def bench(dtype, data_format, func):
  if data_format == 'NHWC':
    x = tf.random.normal((32, 32, 32, 8), dtype=dtype)
    out_grad = tf.random.normal((32, 32, 32, 64), dtype=dtype)
  else:
    x = tf.random.normal((32, 8, 32, 32), dtype=dtype)
    out_grad = tf.random.normal((32, 64, 32, 32), dtype=dtype)
  f = tf.random.normal((3, 3, 8, 8), dtype=dtype)

  p = tf.constant(0.)

  def run():
    if func == 'forward':
      tf.nn.depthwise_conv2d(x, f, [1, 1, 1, 1], 'SAME',
                             data_format=data_format)
    elif func == 'backprop_filter':
      tf.nn.depthwise_conv2d_backprop_filter(
          x, f.shape, out_grad, [1, 1, 1, 1], 'SAME', data_format=data_format)
    else:
      assert func == 'backprop_input'
      tf.nn.depthwise_conv2d_backprop_input(
          x.shape, f, out_grad, [1, 1, 1, 1], 'SAME', data_format=data_format)

  # Warmup
  run()

  start = time.time()
  for _ in range(10):
    run()
  # Synchronize GPU by sending result of computation to CPU
  p = p + 1.
  p.numpy()

  end = time.time()
  print('time for %s %s %s: %s' % (dtype, data_format, func, end - start))

bench('float32', 'NHWC', 'forward')
bench('float32', 'NCHW', 'forward')
bench('float16', 'NHWC', 'forward')
bench('float16', 'NCHW', 'forward')
bench('float32', 'NHWC', 'backprop_input')
bench('float32', 'NCHW', 'backprop_input')
bench('float16', 'NHWC', 'backprop_input')
bench('float16', 'NCHW', 'backprop_input')
bench('float32', 'NHWC', 'backprop_filter')
bench('float32', 'NCHW', 'backprop_filter')
bench('float16', 'NHWC', 'backprop_filter')
bench('float16', 'NCHW', 'backprop_filter')
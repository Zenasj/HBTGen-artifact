from tensorflow import keras
from tensorflow.keras import layers

import argparse
import tensorflow as tf
import timeit

parser = argparse.ArgumentParser(description='TensorFlow bench',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--eager', action='store_true', default=False,
                    help='Enable eager execution')

args = parser.parse_args()
if not args.eager:
    tf.compat.v1.disable_eager_execution()

data = tf.zeros([640, 4096])
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2048),
    tf.keras.layers.Dense(256),
    tf.keras.layers.Dense(10),
])

if args.eager:
    print(timeit.timeit(lambda: model(data), number=100))
else:
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        print(timeit.timeit(lambda: model.predict(data, steps=1), number=100))

import argparse
import tensorflow as tf
import timeit

parser = argparse.ArgumentParser(description='TensorFlow bench',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mode',
                    choices=['eager', 'tf2graph', 'tf1graph'],
                    help='execution mode')

args = parser.parse_args()

if args.mode == "tf1graph":
     tf.compat.v1.disable_eager_execution()

data = tf.zeros([640, 4096])
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2048),
    tf.keras.layers.Dense(256),
    tf.keras.layers.Dense(10),
])

if args.mode == "eager":
    def run_model():
        return model(data)
    run_model()
    print(timeit.timeit(lambda: run_model(), number=1000))
elif args.mode == "tf2graph":
    @tf.function()
    def run_model():
        return model(data)
    run_model()
    print(timeit.timeit(lambda: run_model(), number=1000))
elif args.mode == "tf1graph":
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        model.predict(data, steps=1)
        print(timeit.timeit(lambda: model.predict(data, steps=1), number=1000))
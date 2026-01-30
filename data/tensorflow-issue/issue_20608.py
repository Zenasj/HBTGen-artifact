from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

L = tf.keras.layers # Shorthand

def make_shared_model():
    return tf.keras.Sequential([
        L.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation=tf.nn.relu, input_shape=[28, 28, 1]),
        L.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu),
        L.MaxPool2D(pool_size=(2,2), strides=2)])

def make_model1():
    return tf.keras.Sequential([
        make_shared_model(),
        L.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu),
        L.Flatten(),
        L.Dense(10)])

model1 = make_model1()
# Create a checkpoint with just the shared portion, which is model1.layers[0]
ckpt1 = tf.train.Checkpoint(m = model1.layers[0])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(20):
        # Get your data and train model1
        # Save a checkpoint of the shared components
        pass
    ckpt1.save("/tmp/checkpoint")


# Could have done all this in a single graph, but demonstrating two graphs
# just to be parallel to the code you had above
tf.reset_default_graph()

def make_model2():
    return tf.keras.Sequential([
        make_shared_model(),
        L.Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation=tf.nn.relu),
        L.Flatten(),
        L.Dense(10)])

model2 = make_model2()
# Since model1.layers[0] and model2.layers[0] are isomorphic objects, they can
# save and load checkpoints compatible with each other
ckpt2 = tf.train.Checkpoint(m = model2.layers[0])
status = ckpt2.restore("/tmp/checkpoint")

with tf.Session() as sess:
  status.initialize_or_restore()
  # At this point, the weights for the common portion have been restored
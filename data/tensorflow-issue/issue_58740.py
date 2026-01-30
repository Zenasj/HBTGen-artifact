from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
from pathlib import Path

inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

tf_model_file = Path("model") / "my_model.tf_model"

model.save_weights(tf_model_file, overwrite=True, save_format="tf")

model.load_weights(tf_model_file)
model.save_weights(tf_model_file, overwrite=True, save_format="tf")

import tensorflow as tf
from pathlib import Path
import shutil

inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

tf_model_file = Path("model") / "my_model.tf_model"

model.save_weights(tf_model_file, overwrite=True, save_format="tf")

model.load_weights(tf_model_file)

shutil.rmtree("model")

import tensorflow as tf

ckpt = tf.train.Checkpoint(root=tf.Variable(1.))
path = ckpt.write('my_checkpoint')
ckpt.read(path)
ckpt.write("my_checkpoint")

import tensorflow as tf

ckpt = tf.train.Checkpoint(v=tf.Variable(1.))
path = ckpt.write('my_checkpoint')
ckpt.read(path)
ckpt.write("my_checkpoint")

import tensorflow as tf

inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

ckpt = tf.train.Checkpoint(root=model)
path = ckpt.write('my_checkpoint')
ckpt.read(path)
ckpt.write("my_checkpoint")

import tensorflow as tf

inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

ckpt = tf.train.Checkpoint(model=model)
path = ckpt.write('my_checkpoint')
ckpt.read(path)
ckpt.write("my_checkpoint")
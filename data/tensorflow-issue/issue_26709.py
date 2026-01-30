import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

features = tf.data.Dataset.from_tensors([1.]).repeat(10000).batch(10)
labels = tf.data.Dataset.from_tensors([1.]).repeat(10000).batch(10)
train_dataset = tf.data.Dataset.zip((features, labels))

distribution = tf.contrib.distribute.MirroredStrategy(['/device:CPU:0', '/device:GPU:0', '/device:GPU:1'])

with distribution.scope():
  inputs = tf.keras.layers.Input(shape=(1,))
  predictions = tf.keras.layers.Dense(1)(inputs)
  model = tf.keras.models.Model(inputs=inputs, outputs=predictions)

  model.compile(loss='mean_squared_error',
                optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.2))

model.fit(train_dataset, epochs=5, steps_per_epoch=10)

distribution = tf.contrib.distribute.MirroredStrategy(['/device:CPU:0', '/device:GPU:0', '/device:GPU:1'])

distribution = tf.contrib.distribute.MirroredStrategy(['/device:GPU:0', '/device:GPU:1'])

# Creates a graph.
with tf.device('/cpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
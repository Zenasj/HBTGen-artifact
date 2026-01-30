import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

inputs = tf.keras.layers.Input(shape=(1,))
predictions = tf.keras.layers.Dense(1)(inputs)
model = tf.keras.models.Model(inputs=inputs, outputs=predictions)

features = tf.data.Dataset.from_tensors([1.]).repeat(10000).batch(10)
labels = tf.data.Dataset.from_tensors([1.]).repeat(10000).batch(10)
train_dataset = tf.data.Dataset.zip((features, labels))

distribution = tf.contrib.distribute.MirroredStrategy()

model.compile(loss='categorical_crossentropy',
              optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.2),
              distribute=distribution)
model.fit(train_dataset, epochs=5, steps_per_epoch=10)

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(lr=0.2, momentum=0.9),
              distribute=distribution)
model.fit(train_dataset, epochs=5, steps_per_epoch=10)

model.compile(loss='categorical_crossentropy',
              optimizer=tf.train.AdamOptimizer(learning_rate=0.2),
              distribute=distribution)
model.fit(train_dataset, epochs=5, steps_per_epoch=10)

model.compile(loss='categorical_crossentropy',
              optimizer=tf.train.MomentumOptimizer(learning_rate=0.2, momentum=0.9),
              distribute=distribution)
model.fit(train_dataset, epochs=5, steps_per_epoch=10)

model.compile(loss='categorical_crossentropy',
              optimizer=tf.train.RMSPropOptimzier(learning_rate=0.2),
              distribute=distribution)
model.fit(train_dataset, epochs=5, steps_per_epoch=10)
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_bfloat16')
mixed_precision.set_policy(policy)
inputs = tf.keras.Input(shape=(100,), name='digits')
emb = tf.keras.layers.Embedding(100, 100)
dense = tf.keras.layers.Dense(100)

outputs = emb(inputs) + dense(inputs) # <= the error is here (adding two different types)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
input = tf.random.uniform(shape=[1,25], maxval=100, dtype=tf.int32)
hist = model.fit(input, input, epochs=1, steps_per_epoch=1,verbose=0)
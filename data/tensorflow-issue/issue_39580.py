import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models

a = Input(ragged=True, name='a', shape=(None,), dtype=tf.int32)
b = Input(ragged=True, name='b', shape=(None,), dtype=tf.int32)
embed_a = Embedding(10,  3)(a)
embed_b = Embedding(10, 3)(b)
embed_a_reduced = Lambda(tf.reduce_sum, arguments=dict(axis=1))(embed_a)
embed_b_reduced = Lambda(tf.reduce_sum, arguments=dict(axis=1))(embed_b)
concat = Concatenate()([embed_a_reduced, embed_b_reduced])
out = Dense(1)(concat)
m = Model([a, b], out)
m([tf.ragged.constant([[0], [], [1]]), tf.ragged.constant([[0], [], [1]])])
m.save('/tmp/test')
reloaded_model = tf.keras.models.load_model('/tmp/test')

# embed_a_reduced = Lambda(tf.reduce_sum, arguments=dict(axis=1))(embed_a)
# embed_b_reduced = Lambda(tf.reduce_sum, arguments=dict(axis=1))(embed_b)

embed_a_reduced = Lambda(lambda x:tf.reduce_sum(x,axis=1))(embed_a)
embed_b_reduced = Lambda(lambda x:tf.reduce_sum(x, axis=1))(embed_b)

# This works
embed_a_reduced = tf.reduce_sum(embed_a, axis=1)
embed_b_reduced = tf.reduce_sum(embed_b, axis=1)
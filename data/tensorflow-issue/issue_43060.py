from tensorflow import keras

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as tf_text


print(tf.__version__)
print(tf.keras.__version__)

EMBEDDING = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"

embed = hub.KerasLayer(EMBEDDING, dtype=tf.string, trainable=True)

s1 = tf.keras.Input(shape=[], dtype=tf.string)
s2 = tf.keras.Input(shape=[], dtype=tf.string)

v1 = embed(s1)
v2 = embed(s2)

cd = tf.reduce_sum(tf.multiply(v1, v2), axis=-1)

train_model = tf.keras.Model(inputs=[s1, s2], outputs=[cd])
optimizer = tf.optimizers.SGD(learning_rate=0.001)

i1 = tf.constant(["x", "y", "z"])
i2 = tf.constant(["a", "b", "c"])
c0 = tf.constant([1.0, 1.0, 1.0])

train_model.compile(optimizer=optimizer, loss="mse", metrics=["mse"])
train_model.fit(x=[i1, i2], y=c0, batch_size=1, epochs=5, verbose=2)
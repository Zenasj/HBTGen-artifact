from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

print(tf.version.VERSION)

toy_data = {'movie': [[0], [1], [0], [1]], 'user': [[10], [12], [12], [10]]}
dataset = tf.data.Dataset.from_tensor_slices(toy_data).batch(2)

for x in dataset:
    print(x)

def make_model():
    inp_movie = tf.keras.Input(shape=(1,))
    inp_user = tf.keras.Input(shape=(1,))
    movie_embedding = tf.keras.layers.Dense(
            units=40, activation=tf.keras.layers.Activation("relu"))(inp_movie)
    user_embedding = tf.keras.layers.Dense(
            units=40, activation=tf.keras.layers.Activation("relu"))(inp_user)
    combined = tf.concat([movie_embedding, user_embedding], 1)
    output = tf.keras.layers.Dense(
            units=1, activation=tf.keras.layers.Activation("sigmoid"))(combined)
    model = tf.keras.Model(inputs=[inp_movie, inp_user], outputs=output)
    return model

model = make_model()

for x in dataset:
    print(model(x))
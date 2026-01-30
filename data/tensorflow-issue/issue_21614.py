from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

params = {
    'seq_len': 32,
    'vocab_size': 50,
    'emb_act_size': 32,
    'rnn_act_size': 64,
    'out_size': 16
}


def mfunc(params):
    inputs = tf.keras.layers.Input(shape=(50, ), dtype='int32')
    emb = tf.keras.layers.Embedding(params["vocab_size"],
                                    params["emb_act_size"])(inputs)
    lstm1, s_h, s_c = tf.keras.layers.LSTM(
        params["rnn_act_size"],
        return_sequences=True,
        return_state=True,
        implementation=2)(emb)
    lstm2 = tf.keras.layers.LSTM(
        params["rnn_act_size"], return_sequences=True)(
            lstm1, initial_state=[s_h, s_c])
    y = tf.keras.layers.Dense(params["out_size"], activation='softmax')(lstm2)
    model = tf.keras.Model(inputs, y)
    model.compile(optimizer='sgd', loss='categorical_crossentropy')
    return model

x = mfunc(params)
tf.keras.estimator.model_to_estimator(keras_model=x)
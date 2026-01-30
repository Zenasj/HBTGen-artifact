from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

X = tf.keras.Input([10, 91], name="train_input")
rnn_layers = [
    tf.keras.layers.LSTMCell(size, recurrent_dropout=0, name="rnn_cell%d" % i)
    for i, size in enumerate([512, 512])
]
rnn_output = tf.keras.layers.RNN(rnn_layers, return_sequences=True, name="rnn_layer")(X)
pred_feat = tf.keras.layers.Dense(91, name="prediction_features")(rnn_output)
pred = tf.keras.layers.Softmax()(pred_feat)
model = tf.keras.Model(inputs=[X], outputs=[pred, pred_feat])
model.save("test.h5")
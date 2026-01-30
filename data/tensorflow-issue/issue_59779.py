from tensorflow.keras import layers
from tensorflow.keras import optimizers

from tensorflow import keras

class build_rnn(keras.Model):
  def __init__(self, layer_size = 128, dim = 5, dropout = 0.2):
    super().__init__(self)
    self.pre = keras.Sequential([keras.layers.Input(shape=(None, dim), ragged=True), keras.layers.GaussianNoise(0.2)])

    self.rnn1 = keras.layers.LSTM(layer_size, return_sequences = True, return_state = True, dropout = dropout, kernel_regularizer=l1(1e-6), recurrent_regularizer=l1(1e-6), bias_regularizer=l1(1e-6))
    self.bn = keras.layers.BatchNormalization()
    self.rnn2 = keras.layers.LSTM(layer_size, return_sequences = True, return_state = True, dropout = dropout, kernel_regularizer=l2(1e-6), recurrent_regularizer=l2(1e-6), bias_regularizer=l2(1e-6))
    self.res = keras.layers.Add()
    self.rnn3 = keras.layers.LSTM(layer_size, return_sequences = False, return_state = True, dropout = dropout, kernel_regularizer=l1_l2(1e-6), recurrent_regularizer=l1_l2(1e-6), bias_regularizer=l1_l2(1e-6))
    
    self.dense1 = keras.layers.Dense(32, kernel_regularizer=l1(1e-5), bias_regularizer=l2(1e-5))
    self.bn2 = keras.layers.BatchNormalization()
    self.dense2 = keras.layers.Dense(16, kernel_regularizer=l2(1e-5), bias_regularizer=l1_l2(1e-5))
    self.hosp_out = keras.layers.Dense(1, activation = "sigmoid")

    self.dense4 = keras.layers.Dense(128, kernel_regularizer=l1(1e-5), bias_regularizer=l2(1e-5))
    self.dropout3 = keras.layers.Dropout(0.2)
    self.dense5 = keras.layers.Dense(64, kernel_regularizer=l2(1e-6), bias_regularizer=l1_l2(1e-5))
    self.row_out = keras.layers.Dense(dim, activation = "linear")

    
  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.pre(x, training=training)
    if states is not None: s1, s2, s3 = states
    else: s1 = s2 = s3 = None

    x, h1, s1 = self.rnn1(x, initial_state=s1, training=training)
    xr = self.bn(x, training=training)
    xr, h2, s2 = self.rnn2(xr, initial_state=s2, training=training)
    x = self.res([x, xr])
    x, h3, s3 = self.rnn3(x, initial_state=s3, training=training)

    y = self.dense1(x, training=training)
    y = self.bn2(y, training=training)
    y = self.dense2(y, training=training)
    y = self.hosp_out(y, training=training)

    z = self.dense4(x, training=training)
    z = self.dropout3(z, training=training)
    z = self.dense5(z, training=training)
    z = self.row_out(z, training=training)

    if return_state:
      return y, z, ([h1, s1], [h2, s2], [h3, s3])
      #return x, [h1, s1]
    else:
      return y, z


rmodel = build_rnn(dim = len(columns))

rmodel.compile(loss = ["binary_crossentropy", 'mse'], optimizer=keras.optimizers.experimental.AdamW())

rmodel.fit(x_train, [y_train, z_train], epochs = 100, callbacks=callbacks.EarlyStopping(monitor="loss", patience=7), verbose = 2)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
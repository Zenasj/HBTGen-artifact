from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class BinaryClassificationLSTM(tf.keras.Model):
    def __init__(self, units, name=None):
        super(BinaryClassificationLSTM, self).__init__(name=name)
        self.lstm_layer = tf.keras.layers.LSTM(units, activation='softsign')
        self.dense = tf.keras.layers.Dense(1)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, x):
        sequence = self.lstm_layer(x)
        logit = self.dense(sequence)
        prob = self.sigmoid(logit)

        return prob
    

@tf.function
def get_jacobian(model, tensor_in):
    with tf.GradientTape() as tape:
        tape.watch(tensor_in)
        predictions = model(tensor_in)
    
    return tape.batch_jacobian(predictions, tensor_in)

inp = tf.zeros((1, 10, 5))
lstm = BinaryClassificationLSTM(10)
jacobian = get_jacobian(lstm, inp)
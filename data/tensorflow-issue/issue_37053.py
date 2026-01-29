# tf.random.uniform((1, 10, 5), dtype=tf.float32) ← Inferred input shape from issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define the LSTM-based binary classification model as described
        self.lstm_layer = tf.keras.layers.LSTM(10, activation='softsign')
        self.dense = tf.keras.layers.Dense(1)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, x):
        sequence = self.lstm_layer(x)
        logit = self.dense(sequence)
        prob = self.sigmoid(logit)
        return prob


def my_model_function():
    # Return an instance of MyModel as requested
    return MyModel()


def GetInput():
    # Return a tensor input matching shape and type expected by MyModel:
    # batch size = 1, sequence length = 10, features = 5
    # Using float32 as standard TensorFlow default dtype
    return tf.random.uniform((1, 10, 5), dtype=tf.float32)


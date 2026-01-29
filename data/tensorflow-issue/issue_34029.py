# tf.random.uniform((B, 20, 256), dtype=tf.float32) ← Input shape inferred from code: sequence length 20, feature size 256

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Bidirectional GRU with dropout and recurrent_dropout as in example
        # recurrent_initializer set to glorot_uniform to match original example
        self.bi_gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                units=512,
                dropout=0.2,
                recurrent_dropout=0.2,
                return_sequences=True,
                recurrent_initializer='glorot_uniform'
            )
        )
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.time_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(50))
        self.activation = tf.keras.layers.Activation('softmax')

    def call(self, inputs, training=False):
        x = self.bi_gru(inputs, training=training)
        x = self.dropout(x, training=training)
        x = self.time_dense(x)
        x = self.activation(x)
        return x

def my_model_function():
    # Return a freshly instantiated MyModel; weights uninitialized/random
    return MyModel()

def GetInput():
    # Return a random tensor input with batch size 1 for testing convenience
    # Inputs shape: (batch, time_steps, features) = (1, 20, 256)
    return tf.random.uniform(shape=(1, 20, 256), dtype=tf.float32)


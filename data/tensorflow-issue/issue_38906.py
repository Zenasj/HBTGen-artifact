# tf.random.uniform((1000, 401, 17), dtype=tf.float32)  # Input shape and dtype inferred from original reproduction code

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the architecture from the issue:
        # Conv1D(filters=320, kernel_size=26, activation='relu', input_shape=(401, 17))
        self.conv1d = tf.keras.layers.Conv1D(filters=320, kernel_size=26, activation='relu')
        # MaxPooling1D(pool_size=13, strides=13)
        self.maxpool = tf.keras.layers.MaxPooling1D(pool_size=13, strides=13)
        # Bidirectional GRU(320, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)
        # Note: recurrent_dropout may not be supported on CPU/XLA well, but preserved here for faithfulness
        self.bidirectional_gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(320, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(2000, activation='relu')
        self.dense2 = tf.keras.layers.Dense(301, activation='sigmoid')

    def call(self, inputs, training=False):
        # forward pass mimics the original Sequential model
        x = self.conv1d(inputs)
        x = self.maxpool(x)
        x = self.bidirectional_gru(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Instantiate and compile the model matching the original example
    model = MyModel()
    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Provide a random input tensor matching original input specs (batch=1000, seq_len=401, channels=17)
    return tf.random.uniform((1000, 401, 17), dtype=tf.float32)


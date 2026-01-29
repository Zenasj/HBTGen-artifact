# tf.random.uniform((B, 50, 36), dtype=tf.float32) ← Input shape inferred from Masking layer input_shape=(50,36)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A Masking layer to mask out padding in sequences (input shape: (None, 50, 36))
        self.masking = tf.keras.layers.Masking()

        # Bidirectional LSTM with 128 units and return_sequences=True
        # This was the problematic layer with go_backwards=True due to TFLite reverse op bool support issue
        self.bidirectional = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True)
        )

        # Dropout layer for regularization during training
        self.dropout = tf.keras.layers.Dropout(0.2)

        # Dense output layer with 20 units and softmax activation
        self.dense = tf.keras.layers.Dense(20, activation='softmax')

    def call(self, inputs, training=False):
        x = self.masking(inputs)
        x = self.bidirectional(x)
        x = self.dropout(x, training=training)
        x = self.dense(x)
        return x

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Generates a random float32 tensor matching the input shape expected by MyModel
    # Batch size arbitrarily chosen as 32 here for demonstration
    batch_size = 32
    return tf.random.uniform(shape=(batch_size, 50, 36), dtype=tf.float32)


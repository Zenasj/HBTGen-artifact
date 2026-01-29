# tf.random.uniform((1, 28, 28), dtype=tf.float32) ← Input batch size 1, 28 time steps, 28 features as per LSTM input shape

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Underlying LSTM-based model as described
        self.bidirectional_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(20, return_sequences=True)
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    def call(self, inputs, training=False):
        x = self.bidirectional_lstm(inputs, training=training)
        x = self.flatten(x)
        output = self.dense(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Optionally, set up a build call to initialize weights
    dummy_input = tf.random.uniform([1, 28, 28], dtype=tf.float32)
    model(dummy_input)  # Build model weights
    return model

def GetInput():
    # Batch size 1, 28 time steps, 28 features, dtype matches model input (float32)
    return tf.random.uniform((1, 28, 28), dtype=tf.float32)


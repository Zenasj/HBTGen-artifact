# tf.random.uniform((32, 1, 100), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # LSTM with 200 units, returning only the last output in the sequence
        self.lstm_layer = tf.keras.layers.LSTM(200, return_sequences=False)
        # Dense layer to map to embedding size 100
        self.dense_layer = tf.keras.layers.Dense(100)
    
    def call(self, inputs, training=False):
        x = self.lstm_layer(inputs)
        output = self.dense_layer(x)
        return output

class customLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        # Per-sample loss: mean squared error reduced across last dimension
        # This matches the expected shape to support sample_weight properly.
        return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=-1)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Returns a random tensor matching input shape: (batch_size=32, sequence_len=1, embedding_size=100)
    # Using float32 as typical for embeddings and LSTM input.
    return tf.random.uniform((32, 1, 100), dtype=tf.float32)


# tf.random.uniform((B, T, D), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, TimeDistributed, Dense

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # The original code used a Sequential model with:
        # - Bidirectional LSTM with 128 units, return_sequences=True
        # - TimeDistributed Dense layer with 1 output and relu activation
        #
        # Input shape: (T, D) where T = time steps, D = feature dims
        self.bidirectional_lstm = Bidirectional(LSTM(128, return_sequences=True))
        self.time_distributed_dense = TimeDistributed(Dense(1, activation='relu'))

    def call(self, inputs, training=False):
        # Forward pass matching original model definition
        x = self.bidirectional_lstm(inputs)
        output = self.time_distributed_dense(x)
        return output


def my_model_function():
    # Create and return an instance of MyModel
    return MyModel()


def GetInput():
    # From the original script, input shape is (batch_size, T, D)
    # We do not know T and D specifically, but the training code infers them from data shape:
    # e.g., (_, T, D) = X_train.shape
    #
    # For inference, we pick arbitrary reasonable values for T and D:
    # Let's assume T=10 time steps, D=5 features to generate random input tensor
    batch_size = 64  # The per replica batch size used in the original code
    T = 10
    D = 5
    # Create a random float32 tensor matching input expected by the model
    return tf.random.uniform(shape=(batch_size, T, D), dtype=tf.float32)


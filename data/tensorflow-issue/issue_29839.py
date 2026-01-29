# tf.random.uniform((1, 1024, 7), dtype=tf.float32)  ← inferred input shape based on model input_shape=(window_size=1024, inputs_n=7)

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Parameters inferred from issue:
        self.window_size = 1024
        self.inputs_n = 7
        self.neurons = 128
        self.outputs_n = 4

        # LSTM layers matching the reported model structure
        self.lstm1 = layers.LSTM(self.neurons, return_sequences=True, name="lstm")
        self.lstm2 = layers.LSTM(self.neurons, name="lstm_1")
        self.dense = layers.Dense(self.outputs_n, activation='sigmoid', name="dense")

    def call(self, inputs, training=False):
        """
        Forward pass matching original sequential model:
        Input shape: (batch, 1024, 7)
        Output shape: (batch, 4), activation sigmoid 
        """
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        output = self.dense(x)
        return output


def my_model_function():
    """
    Returns an instance of MyModel.
    The model is uncompiled here; compilation should be done externally if needed.
    """
    model = MyModel()
    return model


def GetInput():
    """
    Generates a random input tensor matching the model's expected input shape:
    shape = (1, 1024, 7), dtype float32 (consistent with feature input).
    """
    return tf.random.uniform(shape=(1, 1024, 7), dtype=tf.float32)


# tf.random.uniform((B, 50, 1), dtype=tf.float32) ← Inferred input shape based on make_dataset and model input reshape

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Reshape, LSTM, RepeatVector, TimeDistributed, Dense

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Parameters inferred from the create_model function & dataset:
        self.input_window_samps = 50
        self.num_signals = 1
        self.output_window_samps = 3
        self.units0 = 10
        self.units1 = 10
        
        # Define layers similarly to create_model()
        self.reshape = Reshape((self.input_window_samps, self.num_signals))
        self.lstm1 = LSTM(self.units0, activation='relu')
        self.repeat = RepeatVector(self.output_window_samps)
        self.lstm2 = LSTM(self.units1, activation='relu', return_sequences=True)
        self.time_dense = TimeDistributed(Dense(self.num_signals))
    
    @tf.function(jit_compile=True)
    def call(self, inputs):
        # inputs shape: (batch_size, input_window_samps * num_signals) => (batch_size, 50*1=50)
        x = self.reshape(inputs)
        x = self.lstm1(x)
        x = self.repeat(x)
        x = self.lstm2(x)
        x = self.time_dense(x)
        return x


def my_model_function():
    # Return an instance of MyModel; no pretrained weights provided – weights will be randomly initialized
    return MyModel()


def GetInput():
    # Return a random input tensor matching expected model input shape:
    # shape = (batch_size, input_window_samps * num_signals)
    # We select batch_size = 32 as typical batch size used in example
    batch_size = 32
    input_window_samps = 50
    num_signals = 1
    
    # Create a random uniform tensor with shape (batch_size, 50 * 1 = 50)
    # Using float32 as default dtype consistent with TensorFlow default and typical model usage
    return tf.random.uniform((batch_size, input_window_samps * num_signals), dtype=tf.float32)


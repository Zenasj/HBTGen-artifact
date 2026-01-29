# tf.random.uniform((B, 180, 1), dtype=tf.float32) ← inferred input shape from training matrix with n_steps=180, n_features=1

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build the model described: Input (180,1) -> BiLSTM(16, return_sequences=True) -> ReLU
        # -> BiLSTM(8, return_sequences=True) -> ReLU -> BiLSTM(16, return_sequences=True) -> ReLU
        # -> TimeDistributed Dense with 1 output feature.
        
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(180,1))
        self.bilstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True))
        self.relu1 = tf.keras.layers.ReLU()
        self.bilstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, return_sequences=True))
        self.relu2 = tf.keras.layers.ReLU()
        self.bilstm3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True))
        self.relu3 = tf.keras.layers.ReLU()
        self.time_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
    
    def call(self, inputs, training=False):
        x = self.input_layer(inputs)
        x = self.bilstm1(x)
        x = self.relu1(x)
        x = self.bilstm2(x)
        x = self.relu2(x)
        x = self.bilstm3(x)
        x = self.relu3(x)
        output = self.time_dense(x)
        return output

def my_model_function():
    # Instantiate the model
    return MyModel()

def GetInput():
    # Generate random input tensor of shape (batch_size, 180, 1)
    # batch_size is arbitrarily chosen to 150 as in the training batch size in the example
    return tf.random.uniform((150, 180, 1), dtype=tf.float32)


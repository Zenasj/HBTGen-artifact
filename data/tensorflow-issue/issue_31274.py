# tf.random.uniform((2048, 100, 300), dtype=tf.float32)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using CuDNNLSTM (or now tf.keras.layers.LSTM with CuDNN enabled by default on GPU)
        self.lstm = tf.keras.layers.LSTM(
            300,
            batch_input_shape=(2048, 100, 300),
            return_sequences=False,
            stateful=False,
            recurrent_activation='sigmoid',  # default in tf.keras is compatible with CuDNN
        )
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.lstm(inputs)
        output = self.dense(x)
        return output

def my_model_function():
    # Instantiate the model
    model = MyModel()
    # Compile with Adadelta optimizer and categorical_crossentropy loss as original code
    optimizer = tf.keras.optimizers.Adadelta()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def GetInput():
    # Return a random input tensor matching the input shape of the model
    # Shape: (batch_size, look_back, feature_count) = (2048, 100, 300)
    # The original numpy data used uniform random floats between -1.0 and 5.0
    input_np = np.random.uniform(low=-1.0, high=5.0, size=(2048, 100, 300)).astype(np.float32)
    # Normally, categorical labels would be integers for classification,
    # but model input only needs the input features tensor.
    return tf.convert_to_tensor(input_np)


# tf.random.uniform((16, 5, 25), dtype=tf.float32) ← inferred input shape from batch_size=16, look_back=5, feature_count=25

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.lstm = tf.keras.layers.LSTM(25, stateful=False)  # Equivalent to CuDNNLSTM in TF 2.x, CuDNNLSTM deprecated
        self.dense = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.lstm(inputs)
        return self.dense(x)

def my_model_function():
    model = MyModel()
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=1.0, epsilon=None)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def GetInput():
    # Return a random tensor input matching (batch_size=16, look_back=5, feature_count=25)
    # Use uniform distribution in range [-1.0, 5], as in original numpy code
    input_shape = (16, 5, 25)
    # dtype float32 for tf model input
    return tf.random.uniform(input_shape, minval=-1.0, maxval=5.0, dtype=tf.float32)


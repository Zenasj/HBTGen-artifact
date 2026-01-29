# tf.random.uniform((4, 1, 2), dtype=tf.float32) ← inferred input shape from get_samples() where X shape is (3, 1, 2) for training; here using batch 4 for example
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # LSTM with 10 units, input shape: (1 timestep, 2 features)
        self.lstm = tf.keras.layers.LSTM(10, input_shape=(1, 2))
        # Output dense layer for 2 coordinates (x,y) with linear activation
        self.dense = tf.keras.layers.Dense(2, activation='linear')

    def call(self, inputs, training=False):
        x = self.lstm(inputs)
        out = self.dense(x)
        return out

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile with MAE loss and Adam optimizer (to match the original script)
    model.compile(loss='mae', optimizer='adam')
    return model

def GetInput():
    # Generate a random batch of 4 samples with shape (time steps=1, features=2)
    # This matches the original data shape from get_samples(): (3,1,2) for training; here batch=4
    return tf.random.uniform((4, 1, 2), dtype=tf.float32)


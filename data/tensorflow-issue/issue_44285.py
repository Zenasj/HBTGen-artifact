# tf.random.uniform((B, None, 5), dtype=tf.float32) ← The input has variable time steps (B=batch, None=timesteps, 5 features: year, month, day, hour, label)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # LSTM layer with 200 units, accepting variable time steps and 5 features
        self.lstm = tf.keras.layers.LSTM(200, input_shape=(None, 5))
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense1 = tf.keras.layers.Dense(100)
        self.dense2 = tf.keras.layers.Dense(1)
        
    def call(self, inputs, training=False):
        x = self.lstm(inputs)
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    import numpy as np
    # Generate a batch of inputs with shape (batch_size, timesteps, features)
    # Let's assume batch_size = 4, timesteps = 24 (like the window in the original dataset), features=5
    batch_size = 4
    timesteps = 24
    # Features: year, month, day, hour, label - we generate dummy plausible data with float dtype
    # year: 2012 to 2014 range, month: 1-12, day: 1-31, hour: 0-23, label: arbitrary float
    year = np.random.randint(2012, 2015, size=(batch_size, timesteps, 1)).astype(np.float32)
    month = np.random.randint(1, 13, size=(batch_size, timesteps, 1)).astype(np.float32)
    day = np.random.randint(1, 29, size=(batch_size, timesteps, 1)).astype(np.float32)  # simplified 28 days for safety
    hour = np.random.randint(0, 24, size=(batch_size, timesteps, 1)).astype(np.float32)
    label = np.zeros((batch_size, timesteps, 1), dtype=np.float32)  # mimic the feature column 'label' zeroed out as in the original code
    
    input_data = np.concatenate([year, month, day, hour, label], axis=-1)
    # Convert to tf.Tensor of float32 (matching model expectations)
    return tf.convert_to_tensor(input_data, dtype=tf.float32)


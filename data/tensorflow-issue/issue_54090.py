# tf.random.uniform((B, 1, look_back), dtype=tf.float32) ← Assumed input shape (batch_size, timesteps=1, features=look_back)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, look_back=1):
        super().__init__()
        # LSTM layer with 100 units, return sequences True, input shape (1, look_back)
        self.lstm = tf.keras.layers.LSTM(
            100, return_sequences=True, input_shape=(1, look_back)
        )
        # Dense layers as per the original model
        self.dense1 = tf.keras.layers.Dense(30, activation='relu')
        self.dense2 = tf.keras.layers.Dense(20, activation='relu')
        self.dense3 = tf.keras.layers.Dense(10, activation='relu')
        self.dense4 = tf.keras.layers.Dense(1)
    
    def call(self, inputs, training=False):
        x = self.lstm(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        # Output shape: (batch_size, timesteps=1, 1)
        return x

def my_model_function():
    # Return an instance of the model with default look_back=1
    model = MyModel(look_back=1)
    # Compilation according to original code with ExponentialDecay learning rate and Adam optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-5,
        decay_steps=10000,
        decay_rate=0.99
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def GetInput():
    # Return a random input tensor with shape (batch_size, timesteps=1, features=look_back=1)
    # Using batch_size=4 for example; dtype float32 as typical for TF models
    batch_size = 4
    look_back = 1  # matches the model's expected input feature dimension
    return tf.random.uniform((batch_size, 1, look_back), dtype=tf.float32)


# tf.random.uniform((batch_size, 20), dtype=tf.float32) ← input shape inferred from model input_shape=(20,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define layers similar to original Sequential model in the issue
        self.dense1 = tf.keras.layers.Dense(50, activation=None)
        self.dense2 = tf.keras.layers.Dense(60, activation=None)
        self.dense3 = tf.keras.layers.Dense(60, activation=None)
        self.dense4 = tf.keras.layers.Dense(60, activation=None)
        self.dense5 = tf.keras.layers.Dense(60, activation=None)
        self.dense6 = tf.keras.layers.Dense(1, activation=None)
        
    def call(self, inputs, training=False):
        # Forward pass similar to Sequential model
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)
        return x

def my_model_function():
    # Instantiate and return the model
    model = MyModel()
    # Compile model to align with the example in the issue and enable usage similar to original code
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

def GetInput():
    # Return a random batch input tensor compatible with model input shape
    # Assuming batch size of 32 for general testing purpose
    batch_size = 32
    input_tensor = tf.random.uniform((batch_size, 20), dtype=tf.float32)
    return input_tensor


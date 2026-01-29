# tf.random.uniform((B, 169, 7), dtype=tf.float64) ← Inferred input shape after transpose, matching model input_dim=7 on last axis

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the same architecture as given:
        # First Dense expects input_dim=7, so last dimension must be 7
        self.dense1 = tf.keras.layers.Dense(12, activation='relu', input_shape=(7,))
        self.dense2 = tf.keras.layers.Dense(8, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=False):
        # inputs shape: (batch_size, 169, 7)
        # We'll process each of the 169 timesteps independently with Dense layers
        # So reshape inputs to (batch*169, 7), apply Denses, then reshape back or reduce
        batch_size = tf.shape(inputs)[0]
        timesteps = tf.shape(inputs)[1]

        x = tf.reshape(inputs, (-1, 7))  # flatten timesteps dimension
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)  # shape (batch*timesteps, 1)

        # reshape back to (batch, timesteps, 1)
        x = tf.reshape(x, (batch_size, timesteps, 1))
        # reduce along timesteps dimension to get final prediction per sample
        # Here, we take mean, but in real case could be max or other aggregation
        x = tf.reduce_mean(x, axis=1)
        # x shape: (batch_size, 1)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Batch size is not specified, choose an arbitrary batch size (e.g., 10)
    # dtype should match original data, float64
    # Input shape expected by model: (batch, 169, 7)
    batch_size = 10
    return tf.random.uniform((batch_size, 169, 7), dtype=tf.float64)


# tf.random.uniform((B, T, F), dtype=tf.float32) ← Assumed shape: (batch, time_steps, features)

import tensorflow as tf
from tensorflow.keras import layers, models

class MyModel(tf.keras.Model):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        # Build a sequential-like structure inside this custom model
        self.lstm1 = layers.LSTM(64, return_sequences=True, input_shape=input_shape)
        self.lstm2 = layers.LSTM(16, activation='relu')
        self.dense = layers.Dense(output_shape)
    
    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        output = self.dense(x)
        return output

def my_model_function():
    # Assumptions based on the original code:
    # - input_shape: (history_length, series_features_count)
    # - output_shape: forecast_length (integer)
    # We'll pick some default values for demo purposes.
    # These should be replaced by the user's real problem dimensions.
    input_shape = (20, 10)   # e.g. 20 timesteps, 10 features
    output_shape = 5         # e.g. forecasting 5 steps ahead

    model = MyModel(input_shape, output_shape)
    
    # Custom metric: R2 implemented in TF graph-friendly way
    # This avoids numpy calls inside metrics and uses TF ops only.
    def r2(y_true, y_pred):
        total_error = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        unexplained_error = tf.reduce_sum(tf.square(y_true - y_pred))
        r_squared = 1.0 - (unexplained_error / (total_error + 1e-7))
        return r_squared

    # Compile with run_eagerly=False (default) as metric is TF graph friendly
    model.compile(optimizer='adam', loss='mse', metrics=[r2])
    return model

def GetInput():
    # Matches the assumed input shape for MyModel instance above
    # batch size 8 is arbitrary — any batch size would work
    batch_size = 8
    time_steps = 20
    features = 10
    # Generate a random tensor of shape (batch_size, time_steps, features)
    return tf.random.uniform(shape=(batch_size, time_steps, features), dtype=tf.float32)


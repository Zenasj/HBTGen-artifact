# tf.random.uniform((B, 100, 4), dtype=tf.float32)  # Input shape inferred from original example: batch size B, sequence length 100, feature size 4

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # LSTM layer with 50 units, returning sequences (shape [B, 100, 50])
        self.lstm = tf.keras.layers.LSTM(units=50, return_sequences=True)
        # LayerNormalization layer (the layer reportedly causes TPU issues in TF 2.3)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        # Global average pooling on temporal dimension
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last')
        # Final Dense layer with 7 output classes, softmax activation
        self.classifier = tf.keras.layers.Dense(7, activation='softmax')

    def call(self, inputs, training=False):
        x = self.lstm(inputs)
        x = self.layer_norm(x)
        x = self.global_avg_pool(x)
        return self.classifier(x)


def my_model_function():
    # Return an instance of the model, no pretrained weights provided
    return MyModel()


def GetInput():
    # Return a random tensor input matching the expected input shape:
    # Shape: (batch_size=32, sequence_length=100, features=4)
    # Using batch size 32 because smaller batch sizes showed issues with LayerNormalization on TPU in the original issue.
    # Input dtype: float32, per TPU-supported types.
    batch_size = 32
    sequence_length = 100
    feature_size = 4
    return tf.random.uniform(
        shape=(batch_size, sequence_length, feature_size),
        minval=-1.0,
        maxval=1.0,
        dtype=tf.float32
    )


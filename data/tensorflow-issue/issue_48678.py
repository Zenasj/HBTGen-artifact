# tf.random.uniform((B, None, 11, 11, 1), dtype=tf.float32) ← input shape (batch, timesteps, height, width, channels)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Masking layer to mask timesteps where all pixels are zero
        self.masking = tf.keras.layers.Masking(mask_value=0.0)
        # ConvLSTM2D layer with 64 filters, 3x3 kernel, relu activation, padding same
        self.convlstm = tf.keras.layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            return_sequences=False)  # default return last output
        # MaxPooling2D layer with 2x2 pool size
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        # inputs shape assumed (batch, time, height, width, channels)
        x = self.masking(inputs)
        # ConvLSTM2D expects time major or batch major with 5D input: (batch, time, height, width, channels)
        x = self.convlstm(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Generates a random tensor shaped (batch=64, timesteps=300, height=11, width=11, channels=1)
    # Matches example from the issue:
    #   np.random.rand(64, 300, 11, 11, 1)
    # Using float32 dtype for TensorFlow compatibility
    return tf.random.uniform(shape=(64, 300, 11, 11, 1), dtype=tf.float32)


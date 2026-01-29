# tf.random.uniform((B, 6, 108, 192, 3), dtype=tf.float32) ← batch inputs are 6-frame sequences of 108x192 RGB images

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # CNN sub-model that processes single images (108,192,3)
        inputs = tf.keras.Input(shape=(108, 192, 3), name='img')
        x = layers.Conv2D(16, 3, activation='relu')(inputs)
        x = layers.Conv2D(16, 3, activation='relu')(x)
        block_1_output = layers.MaxPooling2D(2)(x)

        x = layers.Conv2D(16, 3, activation='relu', padding='same')(block_1_output)
        x = layers.add([x, block_1_output])
        block_2_output = layers.MaxPooling2D(2)(x)

        x = layers.Conv2D(16, 3, activation='relu', padding='same')(block_2_output)
        x = layers.add([x, block_2_output])
        x = layers.MaxPooling2D(2)(x)
        block_3_output = layers.GlobalAveragePooling2D()(x)

        self.cnn = tf.keras.Model(inputs, block_3_output, name='toy_resnet_cnn')

        # The sequence model that applies the CNN on each frame via TimeDistributed
        self.time_distributed = layers.TimeDistributed(self.cnn)

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(16, activation='relu')
        self.dense2 = layers.Dense(1)
        # final activation softplus to ensure output positivity
        self.final_activation = tf.keras.activations.softplus

    def call(self, inputs, training=None):
        # inputs shape: (batch_size, 6, 108, 192, 3)
        x = self.time_distributed(inputs)   # shape: (batch_size, 6, cnn_feature_dim)
        x = self.flatten(x)                  # shape: (batch_size, 6 * cnn_feature_dim)
        x = self.dense1(x)                   # shape: (batch_size, 16)
        x = self.dense2(x)                   # shape: (batch_size, 1)
        counts = self.final_activation(x)   # shape: (batch_size, 1)
        return counts

def my_model_function():
    # Return an instance of MyModel()
    return MyModel()

def GetInput():
    # Returns a random tensor simulating a batch of 10 samples,
    # each with 6 frames of 108x192 RGB images, dtype float32
    return tf.random.uniform(shape=(10,6,108,192,3), dtype=tf.float32)


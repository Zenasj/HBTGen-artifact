# tf.random.uniform((B, 1280, 1), dtype=tf.float32) ← Based on input_shape=(window_size, 1), window_size seems to be 1280

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the original architecture from the post
        # The input shape is (window_size, 1), assuming window_size=1280 (deduced from layer output shapes and conv layer kernels)
        
        self.conv1 = tf.keras.layers.Conv1D(filters=16, kernel_size=2, activation='relu')
        self.conv2 = tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu')
        self.conv3 = tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu')
        self.conv4 = tf.keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu')
        self.maxpool = tf.keras.layers.MaxPooling1D(pool_size=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(4, activation='softmax')  # cats=4 from model summary

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        logits = self.dense(x)
        return logits


def my_model_function():
    # Return an instance of MyModel, no pretrained weights specified in the issue
    return MyModel()


def GetInput():
    # Based on the conv1d input_shape=(window_size, 1) where window_size is inferred to be 1280 (from output shapes)
    # Batch size B can be arbitrary, let's choose B=32 as a reasonable default batch size
    B = 32
    W = 1280
    C = 1
    # Generate random float32 input consistent with usage on TPU and model input requirements
    return tf.random.uniform((B, W, C), dtype=tf.float32)


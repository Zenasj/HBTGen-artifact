# tf.random.uniform((64, 16000, 1), dtype=tf.float32) ← inferred input shape from dataset batch_size=64, audio_len=16000, channels=1

import numpy as np
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.batch_size = 64
        self.audio_len = 16000

        # map_model: a small 1D Conv model for audio sequence
        self.map_model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, 3, padding='same', name='conv1d_1'),
            tf.keras.layers.Conv1D(1, 3, padding='same', name='conv1d_2'),
        ])

        # aux_model: Dense layer, not used in computation but included to replicate scenario
        self.aux_model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, name='dense')
        ])
        self.aux_model.trainable = False  # replicates the issue scenario

        self.global_step = 0

    def call(self, inputs, training=True):
        # Forward pass just through map_model as in original
        return self.map_model(inputs)

    def train_step(self, data):
        mixed_audio, clean_audio = data

        with tf.GradientTape() as tape:
            decoded_audio = self.map_model(mixed_audio)
            total_loss = tf.reduce_mean(tf.abs(clean_audio - decoded_audio))

        grads = tape.gradient(total_loss, self.trainable_variables)

        # Debug: print standard deviation of gradients on conv1d layers
        # Just printing for conv1d* named variables (typical Keras naming pattern)
        for g, tv in zip(grads, self.trainable_variables):
            if 'conv1d' in tv.name and g is not None:
                tf.print(f'Gradient std for {tv.name}:', tf.math.reduce_std(g))

        # Apply the computed gradients to update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.global_step += 1

        return {'loss': total_loss}


def my_model_function():
    # Return an instance of MyModel as requested
    return MyModel()


def GetInput():
    # Generate a tuple of inputs:
    # input tensor shape: (batch_size, audio_len, 1) random floats
    # target tensor shape: same shape as input (clean audio)
    batch_size = 64
    audio_len = 16000
    # Random input and target tensors simulating mixed and clean audio
    mixed_audio = tf.random.uniform((batch_size, audio_len, 1), dtype=tf.float32)
    clean_audio = tf.random.uniform((batch_size, audio_len, 1), dtype=tf.float32)
    return (mixed_audio, clean_audio)


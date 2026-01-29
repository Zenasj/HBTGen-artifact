# tf.random.uniform((B, 84, 84, 3), dtype=tf.float32) ← Assumed input shape based on env observation (84x84 RGB images)

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    """
    A model representing the PPO policy network for RL with CNN processing on image frames.
    
    This is a simplified, self-contained inference model extracted and adapted from
    the context of custom PPO policy 'mnih15' architecture registration seen in the issue.
    
    Input: batch of stacked grayscale or RGB frames resized to 84x84x3 (here assumed 3 channels due to preprocessor)
    Output: policy logits for discrete action space (assumed 9 actions from gym Discrete(9) seen in default_policy)
    
    Notes:
    - Original code used ray.rllib and PPO objects with complex configs.
    - This model implements a small CNN feature extractor + dense layers producing logits.
    - The inference model is stateless and does not include LSTM, though args indicated optional use_lstm=True.
    - For reproducibility, stacked frames as 4 aren't stacked in channel dimension, so shape is (B, 84, 84, 3).
    - Model structure is inspired by classic Atari CNN architectures.
    """
    def __init__(self):
        super().__init__()
        # CNN layers roughly inspired by DeepMind Atari model / mnih15 architecture
        self.conv1 = layers.Conv2D(16, (8, 8), strides=4, activation='relu', padding='valid')
        self.conv2 = layers.Conv2D(32, (4, 4), strides=2, activation='relu', padding='valid')
        self.conv3 = layers.Conv2D(64, (3, 3), strides=1, activation='relu', padding='valid')
        
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(256, activation='relu')
        # Output layer: logits for 9 discrete actions (as per env action space Discrete(9))
        self.logits = layers.Dense(9, activation=None)

    def call(self, inputs, training=False):
        """
        Forward pass
        inputs: Tensor of shape (B, 84, 84, 3) dtype float32 or uint8
        Returns: logits tensor (B, 9)
        """
        x = tf.cast(inputs, tf.float32) / 255.0  # normalize pixel values to [0,1]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        logits = self.logits(x)
        return logits


def my_model_function():
    """
    Returns a new instance of the MyModel.

    This factory method abstracts model initialization.
    """
    return MyModel()


def GetInput():
    """
    Returns a random input tensor that matches the input expected by MyModel.

    Shape: (batch_size=1, height=84, width=84, channels=3)
    dtype: tf.float32 (values scaled 0-255 as typical image tensors)
    """
    # Generate a batch of one random 84x84 RGB frame input with pixel values [0,255]
    input_tensor = tf.random.uniform(shape=(1, 84, 84, 3), minval=0, maxval=256, dtype=tf.float32)
    return input_tensor


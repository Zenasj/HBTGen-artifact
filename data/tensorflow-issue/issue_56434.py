# tf.random.uniform((B, 4, 84, 84), dtype=tf.uint8) ← Input shape and type inferred from Atari DQN example (WINDOW_LENGTH=4, IMAGE 84x84)

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Permute, Conv2D, Flatten, Dense

# The number of actions for the Atari environment, from the example (SpaceInvaders-v0)
nb_actions = 6  # Common for SpaceInvaders-v0, inferred from gym environment

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Following the create_q_model architecture from the issue, adapted as keras.Model subclass layers:
        
        # Input shape expected: (WINDOW_LENGTH=4, 84, 84)
        # We'll implement as a call method expecting inputs shaped (batch, 4, 84, 84)
        # The original model permutes input from (4, 84, 84) -> (84, 84, 4),
        # but the example permutes with (3,2,1), basically inputs (4,84,84) → permute axes to (84,84,4)
        # We'll replicate that.
        
        self.permute = Permute((3, 2, 1))  # Reorder input dims

        self.conv1 = Conv2D(32, kernel_size=8, strides=4, activation='relu')
        self.conv2 = Conv2D(64, kernel_size=4, strides=2, activation='relu')
        self.conv3 = Conv2D(64, kernel_size=3, strides=1, activation='relu')

        self.flatten = Flatten()
        self.dense1 = Dense(256, activation=None)  # No explicit activation after dense in example before final layer
        self.dense2 = Dense(nb_actions, activation='linear')  # Outputs Q-values per action

    def call(self, inputs, training=False):
        """
        Forward pass of the model.

        inputs: tf.Tensor of shape (batch, WINDOW_LENGTH=4, 84, 84), dtype uint8 or float32 normalized
        returns: tf.Tensor of shape (batch, nb_actions), Q-values
        """
        # Expect uint8 input (raw pixel frames), convert to float32 normalized [0,1]
        if inputs.dtype != tf.float32:
            x = tf.cast(inputs, tf.float32) / 255.0
        else:
            x = inputs

        x = self.permute(x)  # Reorder dims to (batch, 84, 84, 4) for Conv2D

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.flatten(x)
        x = self.dense1(x)
        q_values = self.dense2(x)
        return q_values


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Batch size 1, WINDOW_LENGTH (time frames) =4, height=84, width=84, single channel grayscale as uint8 pixel frames
    # Shape inferred from Atari example input (WINDOW_LENGTH, 84, 84)
    # Use uint8 to simulate raw pixel input, consistent with AtariProcessor output
    import tensorflow as tf
    B = 1
    WINDOW_LENGTH = 4
    H, W = 84, 84
    # Input shape: (B, WINDOW_LENGTH, H, W), dtype uint8
    return tf.random.uniform(shape=(B, WINDOW_LENGTH, H, W), minval=0, maxval=256, dtype=tf.int32).astype(tf.uint8)


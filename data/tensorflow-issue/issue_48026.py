import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class TimeDistributedMaskPropagating(tf.keras.layers.TimeDistributed):
    """TimeDistributed layer that propagates mask."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_masking = True
        
    def compute_mask(self, inputs, mask=None):
        return mask

n_features = 3
n_channels = 1

cnn_block = tf.keras.layers.Flatten()
estimator = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(None, n_features, n_channels)),
    tf.keras.layers.Masking(),
    TimeDistributedMaskPropagating(cnn_block),
    # tf.keras.layers.LSTM(10)
    # yields ValueError: Dimensions must be equal, but are 3 and 10
])

x1 = tf.random.uniform((4, 3, 1))       # shape: 4, 3, 1
x2 = tf.random.uniform((3, 3, 1))       # shape: 3, 3, 1

paddings = tf.constant([[0, 1], [0, 0], [0, 0]])
padded_x2 = tf.pad(x2, paddings)        # shape: 4, 3, 1
mini_batch = tf.stack((x1, padded_x2))  # shape: 2, 4, 3, 1

logits = estimator(mini_batch)          # shape: 2, 4, 3
print(logits._keras_mask)               # shape: 2, 4, 3
# mask has shape 2, 4, 3 with values
# [[[ True  True  True]
#   [ True  True  True]
#   [ True  True  True]
#   [ True  True  True]]
# 
#  [[ True  True  True]
#   [ True  True  True]
#   [ True  True  True]
#   [False False False]]]

# mask should have shape 2, 4 with values
# [[ True  True  True  True]
#  [ True  True  True False]]
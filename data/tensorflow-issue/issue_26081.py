# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32) ← Input batch shape for image data

import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet

# This model fuses the described MobileNet-based estimator for gender and age.
# Input: batch of images shape (B, 224, 224, 3), float32 normalized
# Output: tuple of two tensors (gender_probs, age_probs) with shapes (B,2), (B,21)
#
# This matches the original design: MobileNet base, GAP, dropout, dense, then two outputs.

class MyModel(tf.keras.Model):
    def __init__(self, image_size=224, alpha=1.0, num_neu=21, dropout_rate=0.5, fc_layer_size=1024, weights='imagenet'):
        super().__init__()
        self._input_shape = (image_size, image_size, 3)
        self.alpha = alpha
        self.num_neu = num_neu
        self.dropout_rate = dropout_rate
        self.fc_layer_size = fc_layer_size

        # MobileNet base without top layers; will output feature map (7,7,1024) for alpha=1
        self.mobilenet_base = MobileNet(input_shape=self._input_shape,
                                        alpha=alpha,
                                        include_top=False,
                                        weights=weights,
                                        pooling=None)  # no pooling, output (7,7,1024)

        self.global_avg_pool = GlobalAveragePooling2D()
        self.dropout = Dropout(dropout_rate)
        self.dense1 = Dense(fc_layer_size, activation='relu')
        # Two output heads
        self.gender_dense = Dense(2, activation='softmax', name='gender')
        self.age_dense = Dense(num_neu, activation='softmax', name='age')

    def call(self, inputs, training=False):
        x = self.mobilenet_base(inputs)
        x = self.global_avg_pool(x)
        x = self.dropout(x, training=training)
        x = self.dense1(x)

        gender_pred = self.gender_dense(x)
        age_pred = self.age_dense(x)

        return gender_pred, age_pred


def my_model_function():
    # Return a fresh MyModel instance with default weights='imagenet'
    # (Weights can be set None or a filepath in practice.)
    return MyModel()


def GetInput():
    # Generate a batch of 10 random inputs shaped (10,224,224,3), float32 in range [0,1]
    # which matches the expected input for MyModel
    B = 10
    H, W, C = 224, 224, 3
    # Using uniform distribution to simulate inputs; 
    # in practice these would be preprocessed images
    return tf.random.uniform((B, H, W, C), dtype=tf.float32)


# tf.random.uniform((B, 128, 128, 3), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Lambda, Dense, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

# Assumptions based on the issue:
# - Input shape is (128, 128, 3) (typical RGB image size, from generator)
# - Batch size must be specified for the custom RandomHSV layer to avoid None batch dimension errors
# - RandomHSV is an augmentation layer that applies random noise in HSV space during training but passes inputs unchanged during inference
# - The provided model is a Siamese architecture taking two images as input
# - The issue highlighted is about the batch dimension needing to be known for random noise generation
# - The output model here fuses the augmentation and base model logic into a single MyModel to address the issue
# - RandomHSV layer implementation is modified to avoid use of numpy for randomness and to work fully with TF ops to keep it compatible with tf.function jit_compile

class RandomHSV(Layer):
    """
    Layer that adds random HSV noise to RGB images as data augmentation.

    Expects inputs with known batch size (no None batch dimension).
    During training it adds random noise in HSV space per image, else passes input unchanged.

    Args:
      hsv_max_amp: tuple/list of max noise amplitude per HSV channel in range [0,1].
    """
    def __init__(self, hsv_max_amp=(0., 0., 0.), **kwargs):
        super().__init__(**kwargs)
        self.hsv_max_amp = tf.constant(hsv_max_amp, dtype=tf.float32)
        # hsv_max_amp shape: (3,)

    def call(self, inputs, training=None):
        # inputs shape: (batch_size, height, width, 3)
        # Need batch shape fully known for tf.random.uniform shape generation.

        # Defensive: if batch size is unknown, raise (problem in original issue)
        batch_size = tf.shape(inputs)[0]

        def augment():
            # Convert RGB to HSV
            hsv = tf.image.rgb_to_hsv(inputs)  # shape (B,H,W,3), range [0,1]

            # Generate random noise in [-hsv_max_amp, +hsv_max_amp] per channel broadcasted over image
            # Shape: (batch_size,1,1,3) to broadcast across H,W
            random_factors = tf.random.uniform(
                shape=[batch_size, 1, 1, 3],
                minval=-1.0, maxval=1.0,
                dtype=tf.float32
            )
            noise = random_factors * self.hsv_max_amp  # Broadcasting hsv_max_amp (3,) over last dim

            hsv_noisy = tf.clip_by_value(hsv + noise, clip_value_min=0.0, clip_value_max=1.0)
            rgb_noisy = tf.image.hsv_to_rgb(hsv_noisy)
            rgb_noisy.set_shape(inputs.shape)  # maintain static shape information
            return rgb_noisy

        return tf.cond(tf.cast(training, tf.bool),
                       true_fn=augment,
                       false_fn=lambda: inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'hsv_max_amp': self.hsv_max_amp.numpy().tolist()
        })
        return config


def euclidean_distance(embeds):
    # embeds is a list or tuple: [encoded_l, encoded_r]
    (encoded_l, encoded_r) = embeds
    sum_square = tf.reduce_sum(tf.square(encoded_l - encoded_r), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))


def build_body(input_shape, conv2d_filts):
    # A typical convolutional "base" network for encoding individual images in the Siamese net
    # For simplicity, we create a small CNN; original details unknown so inferred minimal example
    inputs = Input(shape=input_shape)
    x = inputs
    for i, filters in enumerate(conv2d_filts):
        x = tf.keras.layers.Conv2D(filters, kernel_size=3, activation='relu', padding='same', name=f'conv_{i}')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2, name=f'maxpool_{i}')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    model = Model(inputs, x, name='embedding_model')
    return model


class MyModel(tf.keras.Model):
    """
    A fused model including:
    - RandomHSV augmentation applied to both inputs (training only)
    - Siamese network body for encoding images
    - Euclidean distance computed on encodings with a sigmoid output for similarity prediction

    Inputs:
      A tuple/list of two images tensors, each shape (batch_size, 128,128,3)
    
    Output:
      Similarity score in [0,1], shape (batch_size, 1)
    """
    def __init__(self):
        super().__init__(name="MySiameseModel")

        # We fix input shape with batch size None, height,width,... must be fixed to avoid None spatial dims
        self.input_shape_single = (128, 128, 3)
        self.conv2d_filts = [32, 64, 128]

        # Augmentation Sequential
        # Using only RandomHSV here as it was custom layer problematic in issue.
        # Original code had: RandomContrast and RandomBrightnessLayer, but not provided - omitted for clarity.
        self.augmentations = RandomHSV(hsv_max_amp=(0.05, 0.25, 0.0), name='RandomHSVPreprocessor')

        # Body network to encode images
        self.body = build_body(self.input_shape_single, self.conv2d_filts)

        # Distance and prediction layers
        self.distance_layer = Lambda(lambda embeds: euclidean_distance(embeds), name='Distance')
        self.prediction_layer = Dense(1, activation='sigmoid', name='Prediction')

    def call(self, inputs, training=None, **kwargs):
        # inputs: list or tuple of two tensors (left_input, right_input)
        left_input, right_input = inputs

        # Apply augmentation on each input (only if training)
        left_augmented = tf.cond(tf.cast(training, tf.bool),
                                 lambda: self.augmentations(left_input, training=True),
                                 lambda: left_input)
        right_augmented = tf.cond(tf.cast(training, tf.bool),
                                  lambda: self.augmentations(right_input, training=True),
                                  lambda: right_input)

        # Encode both inputs
        encoded_l = self.body(left_augmented, training=training)
        encoded_r = self.body(right_augmented, training=training)

        # Compute distance and output prediction
        dist = self.distance_layer([encoded_l, encoded_r])
        out = self.prediction_layer(dist)
        return out


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Create a random input matching what MyModel expects:
    # tuple of two tensors, each (batch_size, 128, 128, 3) with values in [0,1]
    batch_size = 32
    shape = (batch_size, 128, 128, 3)
    # Use tf.random.uniform for float32 [0,1] image data simulation
    img1 = tf.random.uniform(shape, minval=0.0, maxval=1.0, dtype=tf.float32)
    img2 = tf.random.uniform(shape, minval=0.0, maxval=1.0, dtype=tf.float32)
    return (img1, img2)


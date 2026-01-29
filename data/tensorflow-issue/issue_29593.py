# tf.random.uniform((8, 299, 299, 3), dtype=tf.float32) ← This matches the batch_size and input image shape used in the input_fn

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Base Xception model without the top layer, pretrained on ImageNet
        self.base_model = tf.keras.applications.Xception(
            input_shape=(299, 299, 3),
            include_top=False,
            weights='imagenet')
        # Global average pooling layer
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        # Final dense output layer with linear activation and he_normal initializer
        self.output_layer = tf.keras.layers.Dense(
            units=1,
            activation='linear',
            kernel_initializer='he_normal')
        # Freeze the base model initially
        self.base_model.trainable = False

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=training)
        x = self.global_avg_pool(x)
        x = self.output_layer(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a batch of random images with shape (8, 299, 299, 3)
    # Matching batch_size=8 and input image size 299x299 RGB float32 inputs
    # Values normalized to [-1, 1] since original preprocess divides by 127.5 and subtracts 1
    batch_size = 8
    height, width, channels = 299, 299, 3
    # uniform random floats in [0, 1), scale to [-1, 1]
    x = tf.random.uniform(
        shape=(batch_size, height, width, channels),
        minval=0, maxval=1, dtype=tf.float32)
    x = (x * 2.0) - 1.0  # Normalize to [-1,1]
    return x


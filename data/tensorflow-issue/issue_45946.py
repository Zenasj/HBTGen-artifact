# tf.random.uniform((32, 256, 256, 3), dtype=tf.float32) ← Input shape inferred from BATCH_SIZE=32 and IMAGE_SHAPE=[256,256,3]

import tensorflow as tf
from tensorflow import keras

IMAGE_SHAPE = [256, 256, 3]
BATCH_SIZE = 32

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Generator: simple Conv2D layer with output channels = 3 (image channels), no bias, padding same 
        self.generator = keras.Sequential([
            keras.layers.Conv2D(filters=IMAGE_SHAPE[-1], kernel_size=3, strides=1, padding="same", use_bias=False,
                               input_shape=IMAGE_SHAPE)
        ])
        
        # Discriminator: Flatten + Dense output with sigmoid activation
        self.discriminator = keras.Sequential([
            keras.layers.Flatten(input_shape=IMAGE_SHAPE),
            keras.layers.Dense(1, activation="sigmoid")
        ])
        
        # Loss and optimizer
        self.loss_obj = keras.losses.MeanSquaredError()
        self.disc_optimizer = keras.optimizers.Adam(0.0002)

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        """
        Forward pass that fuses generator and discriminator and outputs the discriminator output 
        on generator-produced fake images.
        
        Args:
            inputs: Tensor of images with shape [BATCH_SIZE, 256, 256, 3]
            training: bool for discriminator training mode
            
        Returns:
            disc_fakes: discriminator score for fakes generated from inputs
        """
        fakesA = self.generator(inputs, training=False)
        disc_fakesA = self.discriminator(fakesA, training=training)
        return disc_fakesA
    
    @tf.function(jit_compile=True)
    def train_step(self):
        """
        Performs one training step: generates fake images from inputs, 
        computes discriminator loss, and updates discriminator weights.
        """
        imagesA = tf.random.uniform([BATCH_SIZE] + IMAGE_SHAPE)
        imagesB = tf.random.uniform([BATCH_SIZE] + IMAGE_SHAPE)
        
        # Generate fake images from imagesB with generator
        fakesA = self.generator(imagesB, training=False)
        
        with tf.GradientTape(persistent=True) as tape:
            # Discriminator prediction on fake images
            disc_fakesA = self.discriminator(fakesA, training=True)
            # Discriminator loss against label zeros (fake)
            discA_loss = self.loss_obj(tf.zeros_like(disc_fakesA), disc_fakesA)
        
        # Compute gradients and apply to discriminator weights
        gradients_discA = tape.gradient(discA_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients_discA, self.discriminator.trainable_variables))
        
        return discA_loss

def my_model_function():
    """
    Returns an instance of MyModel, ready for training or inference.
    """
    return MyModel()

def GetInput():
    """
    Returns a random input tensor that matches the model input shape:
    A batch of 32 images, each 256x256 pixels with 3 color channels,
    values in float32 range [0, 1).
    """
    return tf.random.uniform((BATCH_SIZE, *IMAGE_SHAPE), dtype=tf.float32)


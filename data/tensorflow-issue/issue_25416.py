# tf.random.uniform((B, 100), dtype=tf.float32) ← Input is noise vector for generator: batch size B, latent dim 100

import tensorflow as tf
from functools import partial

# Since the original code references "self.dims", "self.channels", "self.img_shape"
# We will fix these as constants here for the reconstructed model.

# Assumptions based on typical MNIST-like WGAN-GP:
DIMS = 64
CHANNELS = 1
IMG_SHAPE = (28, 28, CHANNELS)
LATENT_DIM = 100

# We recreate the gradient penalty loss placeholder (implementation omitted as it was not provided):
def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    # Placeholder implementation that returns zero.
    # In a real WGAN-GP, this should compute the gradient penalty on averaged_samples.
    return tf.constant(0.0)

# Placeholder for RandomWeightedAverage layer used to compute interpolated images:
class RandomWeightedAverage(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        real, fake = inputs
        batch_size = tf.shape(real)[0]
        alpha = tf.random.uniform((batch_size, 1, 1, 1), 0.0, 1.0)
        return alpha * real + (1 - alpha) * fake

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Build generator model
        self.generator = self._build_generator()

        # Build discriminator model
        self.discriminator = self._build_discriminator()

        # Instantiate RandomWeightedAverage layer
        self.weighted_average = RandomWeightedAverage()

        # Create inputs to the combined model analogous to discriminator_model in original code
        # Two inputs: real_img and z_disc (noise)
        self.real_img_input = tf.keras.Input(shape=IMG_SHAPE)
        self.z_input = tf.keras.Input(shape=(LATENT_DIM,))

        # Generate fake image from noise
        self.fake_img = self.generator(self.z_input)

        # Discriminator output for fake and real images
        self.fake_validity = self.discriminator(self.fake_img)
        self.real_validity = self.discriminator(self.real_img_input)

        # Interpolated weighted average image
        self.interpolated_img = self.weighted_average([self.real_img_input, self.fake_img])

        # Discriminator output for interpolated image
        self.interpolated_validity = self.discriminator(self.interpolated_img)

        # Construct Keras model reflecting discriminator_model combining inputs and outputs
        self.discriminator_model = tf.keras.Model(
            inputs=[self.real_img_input, self.z_input],
            outputs=[self.real_validity, self.fake_validity, self.interpolated_validity]
        )

        # Partial function for gradient penalty loss with averaged samples
        self.partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=self.interpolated_img)
        self.partial_gp_loss.__name__ = 'gradient_penalty'

    def _build_generator(self):
        inputs = tf.keras.Input(shape=(LATENT_DIM,))

        x = tf.keras.layers.Dense(7 * 7 * DIMS * 4, activation='relu')(inputs)
        x = tf.keras.layers.Reshape((7, 7, DIMS * 4))(x)

        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Conv2D(DIMS * 4, kernel_size=5, padding="same")(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Conv2D(DIMS * 2, kernel_size=5, padding="same")(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        x = tf.keras.layers.Activation("relu")(x)

        # Uncommented original code commented out here (one more Upsampling step was omitted)

        x = tf.keras.layers.Conv2D(CHANNELS, kernel_size=5, padding="same")(x)
        x = tf.keras.layers.Activation("tanh")(x)

        return tf.keras.Model(inputs, x)

    def _build_discriminator(self):
        inputs = tf.keras.Input(shape=IMG_SHAPE)

        x = tf.keras.layers.Conv2D(DIMS, kernel_size=5, strides=2, padding="same")(inputs)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Conv2D(DIMS * 2, kernel_size=5, strides=2, padding="same")(x)
        x = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Conv2D(DIMS * 4, kernel_size=5, strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Conv2D(DIMS * 4, kernel_size=5, strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)

        return tf.keras.Model(inputs, x)

    def call(self, inputs):
        # Forward pass: expect tuple/list of (real_imgs, noise)
        real_imgs, noise = inputs

        # Generate fake imgs from noise
        fake_imgs = self.generator(noise)

        # Discriminator scores for real and fake imgs
        real_validity = self.discriminator(real_imgs)
        fake_validity = self.discriminator(fake_imgs)

        # Interpolated images between real and fake
        interpolated_imgs = self.weighted_average([real_imgs, fake_imgs])
        interpolated_validity = self.discriminator(interpolated_imgs)

        # Return all three outputs in a tuple to match original model outputs
        return (real_validity, fake_validity, interpolated_validity)

def my_model_function():
    # Instantiate MyModel which bundles generator, discriminator, and combined discriminator model
    return MyModel()

def GetInput():
    # Return a tuple matching the model's call input
    # First item: real images, shape (B, 28, 28, 1), values range ~[-1, 1] as typical for WGAN input tanh output
    # Second item: noise vector, shape (B, 100), uniform random float32
    batch_size = 4  # arbitrary batch size

    real_imgs = tf.random.uniform((batch_size, *IMG_SHAPE), minval=-1.0, maxval=1.0, dtype=tf.float32)
    noise = tf.random.uniform((batch_size, LATENT_DIM), minval=-1.0, maxval=1.0, dtype=tf.float32)
    return (real_imgs, noise)


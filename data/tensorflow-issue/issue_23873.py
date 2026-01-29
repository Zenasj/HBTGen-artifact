# tf.random.normal((B, latent_dimension)) for generator input z
# tf.random.uniform((B, 64, 64, 3), dtype=tf.float32) for encoder/discriminator input x (images)
import tensorflow as tf
from tensorflow import keras as k

KERNEL_INITIALIZER = k.initializers.RandomNormal(mean=0.0, stddev=0.02)


class MyModel(tf.keras.Model):
    """
    Combines the Generator, Encoder, and Discriminator of a BiGAN, and provides outputs
    for training along with the update ops from batch normalization layers.
    
    The forward call returns all outputs needed for the min-max GAN losses so that
    one training step can be constructed outside this model.
    """

    def __init__(self, 
                 visual_shape=(64, 64, 3),
                 latent_dimension=100,
                 output_depth=3,
                 l2_penalty=0.0):
        super().__init__()

        self.latent_dimension = latent_dimension
        self.visual_shape = visual_shape
        self.output_depth = output_depth
        self.l2_penalty = l2_penalty

        # Instantiate submodels
        self.encoder = self._build_encoder()
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

    def _build_discriminator(self):
        kernel_size = (5, 5)

        input_visual = k.layers.Input(shape=self.visual_shape)
        input_encoding = k.layers.Input(shape=(self.latent_dimension,))

        visual = k.layers.Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding="same",
            kernel_initializer=KERNEL_INITIALIZER,
            kernel_regularizer=k.regularizers.l2(self.l2_penalty),
        )(input_visual)
        visual = k.layers.LeakyReLU(alpha=0.1)(visual)

        visual = k.layers.Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=(2, 2),
            padding="same",
            kernel_initializer=KERNEL_INITIALIZER,
            kernel_regularizer=k.regularizers.l2(self.l2_penalty),
        )(visual)
        visual = k.layers.BatchNormalization()(visual)
        visual = k.layers.LeakyReLU(alpha=0.1)(visual)
        visual = k.layers.Dropout(rate=0.5)(visual)

        visual = k.layers.Conv2D(
            filters=64,
            kernel_size=kernel_size,
            strides=(2, 2),
            padding="same",
            kernel_initializer=KERNEL_INITIALIZER,
        )(visual)
        visual = k.layers.BatchNormalization()(visual)
        visual = k.layers.LeakyReLU(alpha=0.1)(visual)
        visual = k.layers.Dropout(rate=0.5)(visual)

        visual = k.layers.Flatten()(visual)

        encoding = k.layers.Dense(units=512, kernel_initializer=KERNEL_INITIALIZER)(
            input_encoding
        )
        encoding = k.layers.LeakyReLU(alpha=0.1)(encoding)
        encoding = k.layers.Dropout(rate=0.5)(encoding)

        mixed = k.layers.Concatenate()([visual, encoding])
        mixed = k.layers.Dense(units=1024, kernel_initializer=KERNEL_INITIALIZER)(mixed)
        mixed = k.layers.LeakyReLU(alpha=0.1)(mixed)
        mixed = k.layers.Dropout(rate=0.5)(mixed)
        features = mixed

        out = k.layers.Dense(1, kernel_initializer=KERNEL_INITIALIZER)(mixed)

        return k.Model(inputs=[input_visual, input_encoding], outputs=[out, features])

    def _build_generator(self):
        kernel_size = (5, 5)
        model = k.Sequential(name="generator")

        model.add(
            k.layers.Dense(
                units=4 * 4 * 64,
                kernel_initializer=KERNEL_INITIALIZER,
                input_shape=(self.latent_dimension,),
                kernel_regularizer=k.regularizers.l2(self.l2_penalty),
            )
        )
        model.add(k.layers.Activation(k.activations.relu))
        model.add(k.layers.Reshape((4, 4, 64)))

        model.add(
            k.layers.Conv2DTranspose(
                filters=64,
                kernel_size=kernel_size,
                strides=(2, 2),
                padding="same",
                kernel_initializer=KERNEL_INITIALIZER,
            )
        )
        model.add(k.layers.BatchNormalization())
        model.add(k.layers.Activation(k.activations.relu))

        model.add(
            k.layers.Conv2DTranspose(
                filters=128,
                kernel_size=kernel_size,
                strides=(2, 2),
                padding="same",
                kernel_initializer=KERNEL_INITIALIZER,
            )
        )
        model.add(k.layers.BatchNormalization())
        model.add(k.layers.Activation(k.activations.relu))

        model.add(
            k.layers.Conv2DTranspose(
                filters=256,
                kernel_size=kernel_size,
                strides=(2, 2),
                padding="same",
                kernel_initializer=KERNEL_INITIALIZER,
            )
        )
        model.add(k.layers.BatchNormalization())
        model.add(k.layers.Activation(k.activations.relu))

        model.add(
            k.layers.Conv2DTranspose(
                filters=self.output_depth,
                kernel_size=kernel_size,
                strides=(2, 2),
                padding="same",
                kernel_initializer=KERNEL_INITIALIZER,
            )
        )
        model.add(k.layers.Activation(k.activations.tanh))  # output in [-1, 1]

        return model

    def _build_encoder(self):
        kernel_size = (5, 5)
        input_visual = k.layers.Input(shape=self.visual_shape)

        visual = k.layers.Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding="same",
            kernel_initializer=KERNEL_INITIALIZER,
            kernel_regularizer=k.regularizers.l2(self.l2_penalty),
        )(input_visual)
        visual = k.layers.LeakyReLU(alpha=0.1)(visual)

        visual = k.layers.Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=(2, 2),
            padding="same",
            kernel_initializer=KERNEL_INITIALIZER,
            kernel_regularizer=k.regularizers.l2(self.l2_penalty),
        )(visual)
        visual = k.layers.BatchNormalization()(visual)
        visual = k.layers.LeakyReLU(alpha=0.1)(visual)
        visual = k.layers.Dropout(rate=0.5)(visual)

        visual = k.layers.Conv2D(
            filters=128,
            kernel_size=kernel_size,
            strides=(2, 2),
            padding="same",
            kernel_initializer=KERNEL_INITIALIZER,
        )(visual)
        visual = k.layers.BatchNormalization()(visual)
        visual = k.layers.LeakyReLU(alpha=0.1)(visual)
        visual = k.layers.Dropout(rate=0.5)(visual)

        visual = k.layers.Flatten()(visual)
        visual = k.layers.Dense(
            units=self.latent_dimension, kernel_initializer=KERNEL_INITIALIZER
        )(visual)

        return k.Model(inputs=input_visual, outputs=visual)

    @tf.function(jit_compile=True)
    def call(self, inputs, training=None):
        """
        inputs: tuple of (x, z)
          x: batch of real images of shape (B, 64, 64, 3), values expected in [-1,1]
          z: batch of latent vectors of shape (B, latent_dimension)

        returns dict with keys:
          'G_z': generated images G(z)
          'E_x': latent encoding of input x
          'G_Ex': reconstructed images through G(E(x))
          'D_Gz': discriminator output for G(z)
          'D_x': discriminator output for x
          'features_Gz': intermediate features from discriminator for G(z)
          'features_x': intermediate features from discriminator for x
          'update_ops': list of batch norm update ops to be executed during training
        """

        x, z = inputs
        # Generate image from random latent vector z
        G_z = self.generator(z, training=training)
        # Encode real image x to latent space
        E_x = self.encoder(x, training=training)
        # Reconstruction through generator from encoded latent
        G_Ex = self.generator(E_x, training=training)

        # Discriminator outputs for G(z) and x
        D_Gz, features_Gz = self.discriminator([G_z, z], training=training)
        D_x, features_x = self.discriminator([x, E_x], training=training)

        # Collect batchnorm update ops from submodels
        # Keras batchnorm layers add update ops on each call, these must be triggered in training session
        update_ops = []
        # Updates from generator
        update_ops += getattr(self.generator, 'updates', [])
        # Updates from encoder
        update_ops += getattr(self.encoder, 'updates', [])
        # Updates from discriminator
        update_ops += getattr(self.discriminator, 'updates', [])

        return {
            'G_z': G_z,
            'E_x': E_x,
            'G_Ex': G_Ex,
            'D_Gz': D_Gz,
            'D_x': D_x,
            'features_Gz': features_Gz,
            'features_x': features_x,
            'update_ops': update_ops,
        }


def my_model_function():
    """
    Returns an instance of the fused BiGAN model
    composed of generator, encoder and discriminator.
    """
    # Using default image size 64x64x3 and latent dim 100 as in the issue
    return MyModel(visual_shape=(64, 64, 3), latent_dimension=100, output_depth=3, l2_penalty=0.0)


def GetInput():
    """
    Generate a random input tuple (x, z) appropriate for MyModel
    x: batch of images with shape (B, 64, 64, 3), float32 in range [-1,1]
    z: batch of latent vectors with shape (B, 100), float32 standard normal approx.

    Batch size B chosen as 4 for example.
    """
    B = 4
    latent_dimension = 100
    # Random images in [-1, 1]
    x = tf.random.uniform((B, 64, 64, 3), minval=-1, maxval=1, dtype=tf.float32)
    # Random latent vectors - standard normal for latent space
    z = tf.random.normal((B, latent_dimension), mean=0.0, stddev=1.0, dtype=tf.float32)
    return (x, z)


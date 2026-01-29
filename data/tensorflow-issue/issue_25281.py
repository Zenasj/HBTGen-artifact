# tf.random.normal((B, 100)) for latent space input z, image input shape (B, 64, 64, 3)
import tensorflow as tf
from tensorflow import keras as k
from typing import Dict
import tensorflow_datasets as tfds

def bce(x: tf.Tensor, label: tf.Tensor, label_smoothing: float = 0.0) -> tf.Tensor:
    """Returns the discrete binary cross entropy between x and the discrete label
    Args:
        x: a 2D tensor
        label: the discrete label, aka, the distribution to match
        label_smoothing: if greater than zero, smooth the labels
    Returns:
        The binary cross entropy
    """
    # Using BinaryCrossentropy from Keras losses
    return k.losses.BinaryCrossentropy()(tf.ones_like(x) * label, x)

def min_max(
    positive: tf.Tensor, negative: tf.Tensor, label_smoothing: float = 0.0
) -> tf.Tensor:
    """Returns the discriminator (min max) loss
    Args:
        positive: the discriminator output for the positive class: 2D tensor
        negative: the discriminator output for the negative class: 2D tensor
        smooth: if greater than zero, applies one-sided label smoothing
    Returns:
        The sum of 2 BCE
    """
    one = tf.constant(1.0)
    zero = tf.constant(0.0)
    d_loss = bce(positive, one, label_smoothing) + bce(negative, zero)
    return d_loss

class Generator(k.Model):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.fc1 = k.layers.Dense(4 * 4 * 1024)
        self.batchnorm1 = k.layers.BatchNormalization()

        self.conv2 = k.layers.Conv2DTranspose(
            filters=512,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
        )
        self.batchnorm2 = k.layers.BatchNormalization()

        self.conv3 = k.layers.Conv2DTranspose(
            filters=256,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
        )
        self.batchnorm3 = k.layers.BatchNormalization()

        self.conv4 = k.layers.Conv2DTranspose(
            filters=128,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
        )
        self.batchnorm4 = k.layers.BatchNormalization()

        self.conv5 = k.layers.Conv2DTranspose(
            filters=3,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
        )
        self.batchnorm5 = k.layers.BatchNormalization()

    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = self.fc1(x)
        x = self.batchnorm1(x, training=training)
        x = tf.nn.relu(x)

        # Reshape dense output into spatial feature map (4x4x1024)
        x = tf.reshape(x, shape=(-1, 4, 4, 1024))

        x = self.conv2(x)
        x = self.batchnorm2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = self.batchnorm3(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv4(x)
        x = self.batchnorm4(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv5(x)
        x = self.batchnorm5(x, training=training)

        x = tf.nn.tanh(x)
        return x

class Discriminator(k.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = k.layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same")
        self.conv2 = k.layers.Conv2D(256, (5, 5), strides=(2, 2), padding="same")
        self.batchnorm2 = k.layers.BatchNormalization()
        self.conv3 = k.layers.Conv2D(512, (5, 5), strides=(2, 2), padding="same")
        self.batchnorm3 = k.layers.BatchNormalization()
        self.conv4 = k.layers.Conv2D(1024, (5, 5), strides=(2, 2), padding="same")
        self.batchnorm4 = k.layers.BatchNormalization()
        self.flatten = k.layers.Flatten()
        self.fc5 = k.layers.Dense(1)

    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = self.conv1(x)
        x = tf.nn.leaky_relu(x)

        x = self.conv2(x)
        x = self.batchnorm2(x, training=training)
        x = tf.nn.leaky_relu(x)

        x = self.conv3(x)
        x = self.batchnorm3(x, training=training)
        x = tf.nn.leaky_relu(x)

        x = self.conv4(x)
        x = self.batchnorm4(x, training=training)
        x = tf.nn.leaky_relu(x)

        x = self.flatten(x)
        x = self.fc5(x)
        return x

class MyModel(tf.keras.Model):
    """Combined model containing Generator and Discriminator
    Provides a callable train_step method for training on real image batches.
    """
    def __init__(self):
        super(MyModel, self).__init__()
        # Instantiate submodules
        self.G = Generator()
        self.D = Discriminator()
        self.latent_vector_dims = 100
        # Optimizers as per original code
        self.G_opt = k.optimizers.Adam(learning_rate=1e-5, beta_1=0.5)
        self.D_opt = k.optimizers.Adam(learning_rate=1e-5, beta_1=0.5)

    @tf.function(jit_compile=True)
    def call(self, inputs, training=True):
        """
        Forward pass returns a dict with generated images and discriminator logits on real and fake.
        This is for demonstration or inference use.
        
        inputs: Either:
            - tensor of latent vectors (noise) to generate images,
            or
            - dict with keys "real_images" and "z" for joint forward of GAN

        returns: dict with keys:
            - 'generated_images': generated images from z
            - 'D_real': discriminator logits on real images
            - 'D_fake': discriminator logits on generated images
        """
        if isinstance(inputs, dict):
            # Expect keys: "real_images", "z"
            real_images = inputs.get("real_images")
            z = inputs.get("z")
            fake_images = self.G(z, training=training)
            D_real = self.D(real_images, training=training)
            D_fake = self.D(fake_images, training=training)
            return {"generated_images": fake_images, "D_real": D_real, "D_fake": D_fake}
        else:
            # Just generate images from noise z input
            generated_images = self.G(inputs, training=training)
            return generated_images

    @tf.function(jit_compile=True)
    def train_step(self, x: tf.Tensor):
        """
        Perform a single GAN train step given a batch of real images x.
        """
        z = tf.random.normal((tf.shape(x)[0], self.latent_vector_dims))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            G_z = self.G(z, training=True)

            D_x = self.D(x, training=True)
            D_Gz = self.D(G_z, training=True)

            g_loss = bce(D_Gz, tf.constant(1.0))
            d_loss = min_max(D_x, D_Gz, label_smoothing=0.0)

        G_grads = gen_tape.gradient(g_loss, self.G.trainable_variables)
        D_grads = disc_tape.gradient(d_loss, self.D.trainable_variables)

        self.G_opt.apply_gradients(zip(G_grads, self.G.trainable_variables))
        self.D_opt.apply_gradients(zip(D_grads, self.D.trainable_variables))

        # Return losses for monitoring
        return d_loss, g_loss


def my_model_function():
    # Return a new instance of MyModel
    return MyModel()

def GetInput():
    """
    Return a random batch input corresponding to real image inputs expected by MyModel.train_step
    The images are 64 x 64 RGB color, values roughly in [0, 1].
    We'll generate a random tensor of shape (B, 64, 64, 3).
    """
    batch_size = 4  # Small batch size for this purpose
    # Random uniform images in [0,1]
    x = tf.random.uniform((batch_size, 64, 64, 3), minval=0.0, maxval=1.0, dtype=tf.float32)
    return x


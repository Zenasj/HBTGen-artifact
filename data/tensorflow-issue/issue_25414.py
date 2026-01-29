# tf.random.normal((32, 100), dtype=tf.float32) ← The GAN latent vector input
import tensorflow as tf
from tensorflow import keras as k


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Generator network
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

        # Discriminator network
        self.d_conv1 = k.layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same")
        self.d_conv2 = k.layers.Conv2D(256, (5, 5), strides=(2, 2), padding="same")
        self.d_batchnorm2 = k.layers.BatchNormalization()
        self.d_conv3 = k.layers.Conv2D(512, (5, 5), strides=(2, 2), padding="same")
        self.d_batchnorm3 = k.layers.BatchNormalization()
        self.d_conv4 = k.layers.Conv2D(1024, (5, 5), strides=(2, 2), padding="same")
        self.d_batchnorm4 = k.layers.BatchNormalization()
        self.d_flatten = k.layers.Flatten()
        self.d_fc5 = k.layers.Dense(1)

    def generator_call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = self.fc1(x)
        x = self.batchnorm1(x, training=training)
        x = tf.nn.relu(x)
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

    def discriminator_call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = self.d_conv1(x)
        x = tf.nn.leaky_relu(x)

        x = self.d_conv2(x)
        x = self.d_batchnorm2(x, training=training)
        x = tf.nn.leaky_relu(x)

        x = self.d_conv3(x)
        x = self.d_batchnorm3(x, training=training)
        x = tf.nn.leaky_relu(x)

        x = self.d_conv4(x)
        x = self.d_batchnorm4(x, training=training)
        x = tf.nn.leaky_relu(x)

        x = self.d_flatten(x)
        x = self.d_fc5(x)
        return x

    def call(self, inputs, training=False):
        """
        Forward method combines generator and discriminator calls.
        
        Assumes inputs is a dict with keys:
            - "latent": latent vector input for generator (shape: (batch, 100))
            - "real_images": real images input to discriminator (optional)
        
        Returns:
            dict with keys:
                - "generated_images": output of generator
                - "discriminator_real": discriminator output on real images
                - "discriminator_fake": discriminator output on generated images
                - "d_loss": discriminator loss (min max)
                - "g_loss": generator loss (bce)
        """
        latent = inputs.get("latent")  # shape (batch, 100)
        real_images = inputs.get("real_images")  # shape (batch, H, W, 3)

        if latent is None or real_images is None:
            raise ValueError("Expected dict input with keys 'latent' and 'real_images'")

        generated_images = self.generator_call(latent, training=training)
        d_real = self.discriminator_call(real_images, training=training)
        d_fake = self.discriminator_call(generated_images, training=training)

        # Compute losses as in the original code:
        g_loss = self.bce(d_fake, 1.0)
        d_loss = self.min_max(d_real, d_fake)

        return {
            "generated_images": generated_images,
            "discriminator_real": d_real,
            "discriminator_fake": d_fake,
            "d_loss": d_loss,
            "g_loss": g_loss,
        }

    @staticmethod
    def bce(x: tf.Tensor, label: float, label_smoothing: float = 0.0):
        """Binary cross entropy with label smoothing."""
        label_tensor = tf.ones_like(x) * label
        return k.losses.BinaryCrossentropy(label_smoothing=label_smoothing)(label_tensor, x)

    @staticmethod
    def min_max(positive: tf.Tensor, negative: tf.Tensor, label_smoothing: float = 0.0):
        """Discriminator min-max loss."""
        d_loss = MyModel.bce(positive, 1.0, label_smoothing) + MyModel.bce(negative, 0.0)
        return d_loss


def my_model_function():
    # Return an instance of MyModel with initialized weights
    return MyModel()


def GetInput():
    # Generate a tuple dict of inputs expected by MyModel:
    # 'latent' is random normal noise vector (batch size 32, dims 100)
    # 'real_images' is a random tensor of shape (batch size 32, H=64, W=64, C=3),
    # consistent with the DCGAN example input image size.
    latent = tf.random.normal((32, 100), dtype=tf.float32)
    real_images = tf.random.uniform((32, 64, 64, 3), minval=-1.0, maxval=1.0, dtype=tf.float32)
    return {"latent": latent, "real_images": real_images}


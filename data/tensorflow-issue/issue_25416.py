import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

generator = gan_generator()
discriminator = gen_discriminator()


# Image input (real sample)
real_img = tf.keras.Input(shape=img_shape)
# Noise input
z_disc = tf.keras.Input(shape=(100,))
# Generate image based of noise (fake sample)
fake_img = generator(z_disc)

# Discriminator determines validity of the real and fake images
fake = discriminator(fake_img)
valid = discriminator(real_img)

# Construct weighted average between real and fake images
interpolated_img = RandomWeightedAverage()([real_img, fake_img])
# Determine validity of weighted sample
validity_interpolated = discriminator(interpolated_img)

# Use Python partial to provide loss function with additional
# 'averaged_samples' argument
partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=interpolated_img)
partial_gp_loss.__name__ = 'gradient_penalty'   # Keras requires function names

discriminator_model = tf.keras.Model(inputs=[real_img, z_disc], outputs=[valid, fake, validity_interpolated])
parallel_discriminator_model = tf.keras.utils.multi_gpu_model(discriminator_model, gpus=2)

def gan_generator(self):
    inputs = tf.keras.Input(shape=(100,))

    x = tf.keras.layers.Dense(7 * 7 * self.dims * 4, activation='relu', input_dim=self.latent_dim)(inputs)
    x = tf.keras.layers.Reshape((7, 7, self.dims * 4))(x)

    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(self.dims * 4, kernel_size=5, padding="same")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(self.dims * 2, kernel_size=5, padding="same")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.Activation("relu")(x)

    # x = UpSampling2D()(x)
    # x = Conv2D(self.dims * 1, kernel_size=5, padding="same")(x)
    # x = BatchNormalization(momentum=0.8)(x)
    # x = Activation("relu")(x)

    x = tf.keras.layers.Conv2D(self.channels, kernel_size=5, padding="same")(x)
    x = tf.keras.layers.Activation("tanh")(x)

    return tf.keras.Model(inputs, x)

def gen_discriminator(self):
    inputs = tf.keras.Input(shape=self.img_shape)

    x = tf.keras.layers.Conv2D(self.dims, kernel_size=5, strides=2, padding="same", input_shape=self.img_shape)(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(self.dims * 2, kernel_size=5, strides=2, padding="same")(x)
    x = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(self.dims * 4, kernel_size=5, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(self.dims * 4, kernel_size=5, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs, x)
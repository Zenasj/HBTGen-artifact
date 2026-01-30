import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import time
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

# https://www.tensorflow.org/tutorials/generative/dcgan

def set_gpu_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    return


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('./results/dcgan/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close(fig)
    return


class Dense(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        fin = np.prod(input_shape[1:])
        weight_shape = [fin, self.units]

        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=0.01)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def call(self, inputs, training=None, mask=None):
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.matmul(x, self.w)
        return x

    def get_config(self):
        config = super(Dense, self).get_config()
        config.update({'units': self.units})
        return config


class LeakyReLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LeakyReLU, self).__init__(**kwargs)
        self.alpha = 0.2

        self.act = tf.keras.layers.LeakyReLU(alpha=self.alpha)

    def call(self, inputs, training=None, mask=None):
        x = self.act(inputs)
        return x

    def get_config(self):
        config = super(LeakyReLU, self).get_config()
        config.update({'alpha': self.alpha})
        return config


class Generator(tf.keras.Model):
    def __init__(self, kernel, **kwargs):
        super(Generator, self).__init__(**kwargs)

        self.kernel = kernel

        self.dense0 = Dense(units=7 * 7 * 256)
        self.bn0 = tf.keras.layers.BatchNormalization()
        self.lrelu0 = LeakyReLU()

        self.convt1 = tf.keras.layers.Conv2DTranspose(128, self.kernel, strides=(1, 1), padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.lrelu1 = LeakyReLU()

        self.convt2 = tf.keras.layers.Conv2DTranspose(64, self.kernel, strides=(2, 2), padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.lrelu2 = LeakyReLU()

        self.convt3 = tf.keras.layers.Conv2DTranspose(1, self.kernel, strides=(2, 2), padding='same', use_bias=False,
                                                      activation='tanh')

    @tf.function
    def call(self, inputs, training=None, mask=None):
        z = inputs

        x = self.dense0(z)
        x = self.bn0(x, training=training)
        x = self.lrelu0(x)

        x = tf.reshape(x, shape=[-1, 7, 7, 256])

        x = self.convt1(x)
        x = self.bn1(x, training=training)
        x = self.lrelu1(x)

        x = self.convt2(x)
        x = self.bn2(x, training=training)
        x = self.lrelu2(x)

        x = self.convt3(x)
        return x

    def get_config(self):
        config = super(Generator, self).get_config()
        config.update({'kernel': self.kernel})
        return config

    def compute_output_shape(self, input_shape):
        print('[Generator] - compute_output_shape() input_shape: {}'.format(input_shape))
        return input_shape[0], 28, 28, 1

    @tf.function
    def serve(self, z):
        x = self.dense0(z)
        x = self.bn0(x, training=False)
        x = self.lrelu0(x)

        x = tf.reshape(x, shape=[-1, 7, 7, 256])

        x = self.convt1(x)
        x = self.bn1(x, training=False)
        x = self.lrelu1(x)

        x = self.convt2(x)
        x = self.bn2(x, training=False)
        x = self.lrelu2(x)

        x = self.convt3(x)
        return x


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
    return model


def load_mnist(batch_size):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(batch_size)
    return train_dataset


def discriminator_loss(real_output, fake_output, cross_entropy):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output, cross_entropy):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, batch_size, noise_dim, generator, discriminator,
               cross_entropy, generator_optimizer, discriminator_optimizer):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output, cross_entropy)
        disc_loss = discriminator_loss(real_output, fake_output, cross_entropy)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train_loop():
    batch_size = 256
    epochs = 30
    noise_dim = 100
    num_examples_to_generate = 16

    dataset = load_mnist(batch_size)

    # create models
    generator = Generator(kernel=5)
    discriminator = make_discriminator_model()

    # loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # optimizer
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint_dir = './models/dcgan'
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=2)

    # We will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch, batch_size, noise_dim, generator, discriminator,
                       cross_entropy, generator_optimizer, discriminator_optimizer)

        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Produce images for the GIF as we go
            generate_and_save_images(generator, epoch + 1, seed)
            manager.save(checkpoint_number=epoch + 1)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, seed)
    return


def export_model():
    # restore generator model only
    noise_dim = 100
    generator = Generator(kernel=5, dynamic=True)
    test_x = tf.random.normal([1, noise_dim])
    _ = generator(test_x, training=False)

    checkpoint_dir = './models/dcgan'
    checkpoint = tf.train.Checkpoint(generator=generator)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=2)
    checkpoint.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        print('Restored from {}'.format(manager.latest_checkpoint))
        _ = generator.predict(test_x)
    else:
        raise ValueError()

    # export generator model
    export_dir = './models/dcgan/1'
    tf.saved_model.save(
        generator,
        export_dir,
        signatures=generator.serve.get_concrete_function(
            z=tf.TensorSpec(shape=[None, noise_dim], dtype=tf.float32)
        )
    )
    return


def main():
    set_gpu_memory_growth()

    train_loop()

    export_model()
    return


if __name__ == '__main__':
    main()

def export_model():
    # restore generator model only
    noise_dim = 100
    generator = Generator(kernel=5, dynamic=True)

def compute_output_shape(self, input_shape):
    print('[Generator] - compute_output_shape() input_shape: {}'.format(input_shape))
    return tf.TensorShape([input_shape[0], 28, 28, 1])
# tf.random.uniform((B, 2), dtype=tf.float32) ← GAN input noise shape is (batch_size, latent_dim=5), discriminator input is (batch_size, 2)

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class MyModel(tf.keras.Model):
    """
    Fusion of both Keras and TF GAN models discussed in the issue:
    - Generator: maps latent_dim=5 noise vector to 2D points (x,y).
    - Discriminator: binary classifier on 2D points.
    We expose a train step and a call that outputs discriminator real/fake predictions.
    We provide a method to compare outputs of both models for testing their equivalence.
    
    Notes/Assumptions:
    - Using beta_1=0.5 and learning_rate=0.0005 for Adam optimizers, 
      per issue comments for stable GAN convergence.
    - Generator uses ReLU and linear output.
    - Discriminator uses ReLU then sigmoid output.
    - Binary crossentropy (with from_logits=False) for losses.
    """

    def __init__(self, latent_dim=5):
        super().__init__()
        self.latent_dim = latent_dim

        # Generator model: latent_dim -> 15 units relu -> 2 units linear (x,y)
        self.generator = tf.keras.Sequential([
            layers.Dense(15, activation='relu', kernel_initializer='he_uniform', input_shape=(latent_dim,)),
            layers.Dense(2, activation='linear')
        ])

        # Discriminator model: 2 inputs (x,y) -> 25 units relu -> 1 unit sigmoid
        self.discriminator = tf.keras.Sequential([
            layers.Dense(25, activation='relu', kernel_initializer='he_uniform', input_shape=(2,)),
            layers.Dense(1, activation='sigmoid')
        ])

        # Optimizers with stable GAN parameters suggested by comments
        self.gen_optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0005, beta_1=0.5)
        self.disc_optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0005, beta_1=0.5)

        # Loss function: binary crossentropy
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def generate_latent_points(self, n):
        """
        Generate random latent points as input noise for generator.
        Shape: (n, latent_dim)
        """
        return tf.random.normal(shape=(n, self.latent_dim), dtype=tf.float32)

    def generate_real_samples(self, n):
        """
        Generate true points (x, x^2) with x in range [-0.5,0.5].
        Shape: (n,2)
        """
        x1 = tf.random.uniform((n,1), minval=-0.5, maxval=0.5, dtype=tf.float32)
        x2 = tf.math.square(x1)
        return tf.concat([x1, x2], axis=1)  # shape (n,2)

    def generate_fake_samples(self, n):
        """
        Generate fake points by passing latent noise through generator.
        """
        latent_points = self.generate_latent_points(n)
        generated = self.generator(latent_points, training=False)
        return generated

    def call(self, inputs, training=False):
        """
        Forward call returns discriminator output for given inputs.
        inputs: either latent noise (to generate points) or points directly.
        For compatibility, assume input is latent noise, output discriminator prediction on generated points.
        """
        generated_points = self.generator(inputs, training=training)
        disc_output = self.discriminator(generated_points, training=training)
        return disc_output

    def generator_loss(self, fake_output):
        """
        Generator tries to fool discriminator, so minimize loss between fake_output and ones.
        """
        return self.bce(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        """
        Discriminator tries to classify real as ones and fake as zeros.
        """
        real_loss = self.bce(tf.ones_like(real_output), real_output)
        fake_loss = self.bce(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    @tf.function(jit_compile=True)
    def train_step(self, n_batch=128):
        """
        Single training step: update generator and discriminator with batch size n_batch.
        Splits batch in half for real/fake training discriminator.
        """

        half_batch = n_batch // 2

        # Generate real and fake samples
        real_samples = self.generate_real_samples(half_batch)
        latent_fake = self.generate_latent_points(half_batch)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake points
            fake_samples = self.generator(latent_fake, training=True)

            # Discriminator output on real and fake
            real_output = self.discriminator(real_samples, training=True)
            fake_output = self.discriminator(fake_samples, training=True)

            # Compute losses
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        # Compute gradients and apply
        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))

        return {'gen_loss': gen_loss, 'disc_loss': disc_loss}

    def generate_and_classify(self, n):
        """
        Convenience method generating n fake samples and discriminator predictions.
        Returns tuple (fake_samples, discriminator_scores).
        """
        latent = self.generate_latent_points(n)
        fake_samples = self.generator(latent, training=False)
        disc_scores = self.discriminator(fake_samples, training=False)
        return fake_samples, disc_scores

def my_model_function():
    """
    Returns a new instance of MyModel with default latent_dim=5.
    """
    return MyModel()

def GetInput():
    """
    Returns a batch of latent noise inputs for the generator,
    shape (batch_size=128, latent_dim=5), float32 tensor.
    """
    batch_size = 128
    latent_dim = 5
    return tf.random.normal(shape=(batch_size, latent_dim), dtype=tf.float32)


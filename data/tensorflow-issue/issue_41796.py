# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from original MNIST domain images

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras.models import Model

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the shared generators used in the domain translation GAN
        self.g_m2h = self._build_generator(name="g_m2h")  # Mapping: domain M to H
        self.g_h2m = self._build_generator(name="g_h2m")  # Mapping: domain H to M
        self.g_h2r = self._build_generator(name="g_h2r")  # Mapping: domain H to R
        self.g_r2h = self._build_generator(name="g_r2h")  # Mapping: domain R to H

        # Define critics (discriminators) for domains M and R
        self.critic_m = self._build_critic(name="critic_m")
        self.critic_r = self._build_critic(name="critic_r")

        # Additional models built over these sub-networks:
        # Cycle consistency model (multitask outputs with cycle losses)
        m_inputs = Input(shape=(28, 28, 1))
        r_inputs = Input(shape=(28, 28, 1))

        # Forward cycle losses for domain M and R
        m_output_1 = self.g_h2m(self.g_m2h(m_inputs))
        m_output_2 = self.g_h2m(self.g_r2h(self.g_h2r(self.g_m2h(m_inputs))))

        r_output_1 = self.g_h2r(self.g_r2h(r_inputs))
        r_output_2 = self.g_h2r(self.g_m2h(self.g_h2m(self.g_r2h(r_inputs))))

        # Cycle consistency outputs explicitly
        mmh = self.g_r2h(self.g_h2r(self.g_m2h(m_inputs)))
        rrh = self.g_m2h(self.g_h2m(self.g_r2h(r_inputs)))

        self.model_cycle = Model(
            inputs=[m_inputs, r_inputs],
            outputs=[m_output_1, m_output_2, r_output_1, r_output_2, mmh, rrh],
            name="cycle_model"
        )

        # Models that use fixed generators (trainable=False) for WGAN GAN training on domain R and M
        # For domain R:
        self.g_m2h.trainable = False
        self.g_h2m.trainable = False
        self.g_h2r.trainable = False
        self.g_r2h.trainable = False

        m_inputs_wgan = Input(shape=(28, 28, 1))
        g_m2r = self.g_h2r(self.g_m2h(m_inputs_wgan))
        self.model_m2r = Model(m_inputs_wgan, g_m2r, name="model_m2r")
        # model_m2r compiled with trainable False sub-networks

        real_image_r = Input(shape=(28, 28, 1))
        noisy_image_r = Input(shape=(28, 28, 1))
        valid_r = self.critic_r(real_image_r)
        fake_image_r = self.model_m2r(noisy_image_r)
        fake_r = self.critic_r(fake_image_r)
        self.model_wgan_r = Model(
            inputs=[real_image_r, noisy_image_r],
            outputs=[valid_r, fake_r],
            name="wgan_critic_r"
        )

        # For domain M:
        r_inputs_wgan = Input(shape=(28, 28, 1))
        g_r2m = self.g_h2m(self.g_r2h(r_inputs_wgan))
        self.model_r2m = Model(r_inputs_wgan, g_r2m, name="model_r2m")

        real_image_m = Input(shape=(28, 28, 1))
        noisy_image_m = Input(shape=(28, 28, 1))
        valid_m = self.critic_m(real_image_m)
        fake_image_m = self.model_r2m(noisy_image_m)
        fake_m = self.critic_m(fake_image_m)
        self.model_wgan_m = Model(
            inputs=[real_image_m, noisy_image_m],
            outputs=[valid_m, fake_m],
            name="wgan_critic_m"
        )

        # Reset trainable flags for further training steps on these sub-networks
        self.g_m2h.trainable = True
        self.g_h2m.trainable = True
        self.g_h2r.trainable = True
        self.g_r2h.trainable = True

        self.critic_r.trainable = False
        m_inputs_critic = Input(shape=(28, 28, 1))
        m2r_critic_out = self.critic_r(self.g_h2r(self.g_m2h(m_inputs_critic)))
        self.model_critic_m2r = Model(m_inputs_critic, m2r_critic_out, name="critic_m2r")

        self.critic_m.trainable = False
        r_inputs_critic = Input(shape=(28, 28, 1))
        r2m_critic_out = self.critic_m(self.g_h2m(self.g_r2h(r_inputs_critic)))
        self.model_critic_r2m = Model(r_inputs_critic, r2m_critic_out, name="critic_r2m")

    @staticmethod
    def _build_generator(name=None):
        # A simple generator: Conv2D with kernel size (3,3), padding=same
        inputs = Input(shape=(28, 28, 1))
        outputs = Conv2D(1, (3, 3), padding='same')(inputs)
        model = Model(inputs, outputs, name=name)
        return model

    @staticmethod
    def _build_critic(name=None):
        # A simple critic/discriminator model: Flatten -> Dense(1)
        inputs = Input(shape=(28, 28, 1))
        x = Flatten()(inputs)
        outputs = Dense(1)(x)
        model = Model(inputs, outputs, name=name)
        return model

    def call(self, inputs, training=False):
        """
        Forward call returns the 6 outputs from the cycle consistency model.
        This illustrates the domain translation pipeline with cycle losses.
        """
        m_input, r_input = inputs  # expecting a tuple of two inputs
        return self.model_cycle([m_input, r_input])

def my_model_function():
    # Return an instance of MyModel with freshly created sub-networks
    return MyModel()

def GetInput():
    # Return a tuple of two random float32 tensors matching the input shape (28, 28, 1)
    # to simulate (m_inputs, r_inputs) for the model call
    x1 = tf.random.uniform((1, 28, 28, 1), dtype=tf.float32)
    x2 = tf.random.uniform((1, 28, 28, 1), dtype=tf.float32)
    return (x1, x2)


# tf.random.uniform((32, 1, 1, 1), dtype=tf.float32) ← inferred alpha shape in RandomWeightedAverage
import tensorflow as tf

class RandomWeightedAverage(tf.keras.layers.Layer):
    """Provides a (random) weighted average between real and generated audio samples"""
    def __init__(self, batch_size=32, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        # inputs is a list of two tensors: [real_audio, fake_audio]
        alpha = tf.random.uniform((self.batch_size, 1, 1, 1), dtype=tf.float32)
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class MyModel(tf.keras.Model):
    def __init__(self, batch_size=32):
        super().__init__()
        # Parameters as given
        self.latent_dim = 100
        self.d = 64
        self.c = 16
        self.a = 1
        self.Fs = 44100
        self.batch_size = batch_size
        self.n_critic = 5

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)

        # Instantiate layers for RandomWeightedAverage with batch_size
        self.random_weighted_average = RandomWeightedAverage(batch_size=self.batch_size)

        # Build the generator and critic models
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        # Compile critic model with gradient penalty loss
        self.generator.trainable = False  # Freeze generator for critic training
        real_audio = tf.keras.Input(shape=(self.a*256*self.d, 1), batch_size=self.batch_size)
        z_disc = tf.keras.Input(shape=(self.a, self.latent_dim), batch_size=self.batch_size)

        fake_audio = self.generator(z_disc)

        valid_real = self.critic(real_audio)
        valid_fake = self.critic(fake_audio)

        interpolated_audio = self.random_weighted_average([real_audio, fake_audio])
        validity_interpolated = self.critic(interpolated_audio)

        # Partial will be replaced by a lambda closure
        def gp_loss(y_true, y_pred):
            return self.gradient_penalty_loss(y_true, y_pred, averaged_samples=interpolated_audio)

        self.critic_model = tf.keras.Model(inputs=[real_audio, z_disc],
                                           outputs=[valid_real, valid_fake, validity_interpolated])
        self.critic_model.compile(optimizer=optimizer,
                                  loss=[self.wasserstein_loss, self.wasserstein_loss, gp_loss],
                                  loss_weights=[1.0, 1.0, 10.0])

        # Compile generator model
        self.critic.trainable = False
        self.generator.trainable = True

        z_gen = tf.keras.Input(shape=(self.a, self.latent_dim), batch_size=self.batch_size)
        gen_audio = self.generator(z_gen)
        valid = self.critic(gen_audio)

        self.generator_model = tf.keras.Model(z_gen, valid)
        self.generator_model.compile(optimizer=optimizer, loss=self.wasserstein_loss)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """Computes gradient penalty based on prediction and weighted real/fake samples"""
        gradients = tf.gradients(y_pred, averaged_samples)[0]
        # Compute L2 norm across each sample
        gradients_sqr = tf.square(gradients)
        gradients_sqr_sum = tf.reduce_sum(gradients_sqr,
                                          axis=tf.range(1, tf.rank(gradients_sqr)))
        gradient_l2_norm = tf.sqrt(gradients_sqr_sum)
        # Penalty for deviation from 1
        gradient_penalty = tf.square(1. - gradient_l2_norm)
        return tf.reduce_mean(gradient_penalty)

    def wasserstein_loss(self, y_true, y_pred):
        return tf.reduce_mean(y_true * y_pred)

    def apply_phaseshuffle(self, x, rad=2, pad_type='reflect'):
        b, x_len, nch = x.shape.as_list()

        phase = tf.random.uniform(shape=[], minval=-rad, maxval=rad+1, dtype=tf.int32)
        pad_l = tf.maximum(phase, 0)
        pad_r = tf.maximum(-phase, 0)
        phase_start = pad_r
        x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

        x = x[:, phase_start:phase_start + x_len]
        x.set_shape([b, x_len, nch])

        return x

    def build_generator(self):
        d = self.d
        c = self.c
        a = self.a

        input_layer = tf.keras.Input(shape=(a, self.latent_dim))

        dense_layer0 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(256 * d, input_shape=(self.latent_dim,))
        )(input_layer)
        reshape_layer0 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Reshape((c, c * d))
        )(dense_layer0)
        relu_layer0 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Activation('relu')
        )(reshape_layer0)

        # WaveCNN Layers with Conv2DTranspose and expansion of dims for TimeDistributed Conv2DTranspose
        c //= 2
        expanded_layer0 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))
        )(relu_layer0)
        conv1d_t_layer0 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2DTranspose(c * d, (1, 25), strides=(1, 4), padding='same')
        )(expanded_layer0)
        slice_layer0 = tf.keras.layers.Lambda(lambda x: x[:, :, 0])(conv1d_t_layer0)
        relu_layer2 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Activation('relu')
        )(slice_layer0)

        c //= 2
        expanded_layer1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))
        )(relu_layer2)
        conv1d_t_layer1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2DTranspose(c * d, (1, 25), strides=(1, 4), padding='same')
        )(expanded_layer1)
        slice_layer1 = tf.keras.layers.Lambda(lambda x: x[:, :, 0])(conv1d_t_layer1)
        relu_layer3 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Activation('relu')
        )(slice_layer1)

        c //= 2
        expanded_layer2 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))
        )(relu_layer3)
        conv1d_t_layer2 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2DTranspose(c * d, (1, 25), strides=(1, 4), padding='same')
        )(expanded_layer2)
        slice_layer2 = tf.keras.layers.Lambda(lambda x: x[:, :, 0])(conv1d_t_layer2)
        relu_layer4 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Activation('relu')
        )(slice_layer2)

        c //= 2
        expanded_layer3 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))
        )(relu_layer4)
        conv1d_t_layer3 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2DTranspose(c * d, (1, 25), strides=(1, 4), padding='same')
        )(expanded_layer3)
        slice_layer3 = tf.keras.layers.Lambda(lambda x: x[:, :, 0])(conv1d_t_layer3)
        relu_layer5 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Activation('relu')
        )(slice_layer3)

        expanded_layer4 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))
        )(relu_layer5)
        conv1d_t_layer4 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2DTranspose(1, (1, 25), strides=(1, 4), padding='same')
        )(expanded_layer4)
        slice_layer4 = tf.keras.layers.Lambda(lambda x: x[:, :, 0])(conv1d_t_layer4)
        tanh_layer0 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Activation('tanh')
        )(slice_layer4)

        reshape_layer1 = tf.keras.layers.Reshape((a * 256 * d, 1))(tanh_layer0)

        model = tf.keras.Model(inputs=input_layer, outputs=reshape_layer1)

        return model

    def build_critic(self):
        d = self.d
        c = self.c
        a = self.a

        input_layer = tf.keras.Input(shape=(a * 256 * d, 1))

        reshape_layer0 = tf.keras.layers.Reshape((a, 256 * d, 1))(input_layer)

        conv1d_layer0 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv1D(d, 25, strides=4, padding='same')
        )(reshape_layer0)
        LReLU_layer0 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.LeakyReLU(alpha=0.2)
        )(conv1d_layer0)
        phaseshuffle_layer0 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Lambda(lambda x: self.apply_phaseshuffle(x))
        )(LReLU_layer0)

        conv1d_layer1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv1D(2 * d, 25, strides=4, padding='same')
        )(phaseshuffle_layer0)
        LReLU_layer1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.LeakyReLU(alpha=0.2)
        )(conv1d_layer1)
        phaseshuffle_layer1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Lambda(lambda x: self.apply_phaseshuffle(x))
        )(LReLU_layer1)

        conv1d_layer2 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv1D(4 * d, 25, strides=4, padding='same')
        )(phaseshuffle_layer1)
        LReLU_layer2 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.LeakyReLU(alpha=0.2)
        )(conv1d_layer2)
        phaseshuffle_layer2 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Lambda(lambda x: self.apply_phaseshuffle(x))
        )(LReLU_layer2)

        conv1d_layer3 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv1D(8 * d, 25, strides=4, padding='same')
        )(phaseshuffle_layer2)
        LReLU_layer3 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.LeakyReLU(alpha=0.2)
        )(conv1d_layer3)
        phaseshuffle_layer3 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Lambda(lambda x: self.apply_phaseshuffle(x))
        )(LReLU_layer3)

        conv1d_layer4 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv1D(16 * d, 25, strides=4, padding='same')
        )(phaseshuffle_layer3)
        LReLU_layer4 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.LeakyReLU(alpha=0.2)
        )(conv1d_layer4)
        phaseshuffle_layer4 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Lambda(lambda x: self.apply_phaseshuffle(x))
        )(LReLU_layer4)

        reshape_layer1 = tf.keras.layers.Reshape((a, 256 * d))(phaseshuffle_layer4)
        slice_layer0 = tf.keras.layers.Lambda(lambda x: x[:, 0])(reshape_layer1)

        dense_layer1 = tf.keras.layers.Dense(1)(slice_layer0)

        model = tf.keras.Model(inputs=input_layer, outputs=dense_layer1)

        return model

def my_model_function():
    # Use a default batch size compatible with typical usage, e.g. 32
    return MyModel(batch_size=32)

def GetInput():
    # Inputs to MyModel are pairs of:
    #  - real_audio: shape (batch_size, a*256*d, 1)
    #  - noise: shape (batch_size, a, latent_dim)
    # According to MyModel constants: a=1, d=64, latent_dim=100
    batch_size = 32
    a = 1
    d = 64
    latent_dim = 100

    real_audio = tf.random.uniform((batch_size, a * 256 * d, 1), dtype=tf.float32)
    noise = tf.random.normal((batch_size, a, latent_dim), dtype=tf.float32)

    # The model as defined expects inputs to critic_model as tuple/list of real_audio and noise
    return [real_audio, noise]


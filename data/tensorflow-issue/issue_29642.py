# tf.random.uniform((B, imsize, imsize, color_channels), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, imsize=28, color_channels=1, intermediate_dim=64, latent_dim=32):
        super().__init__()
        self.imsize = imsize
        self.color_channels = color_channels
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim

        # Encoder layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense_enc = tf.keras.layers.Dense(self.intermediate_dim, activation='relu')
        self.z_mean_layer = tf.keras.layers.Dense(self.latent_dim, name='z_mean')
        self.z_log_var_layer = tf.keras.layers.Dense(self.latent_dim, name='z_log_var')

        # Sampling layer as a submodule
        self.sampling = self.Sampling()

        # Decoder layers
        self.dense_dec = tf.keras.layers.Dense(self.intermediate_dim, activation='relu')
        # Output dim = imsize * imsize * color_channels
        self.dense_output = tf.keras.layers.Dense(self.imsize * self.imsize * self.color_channels, activation='sigmoid')
        self.reshape_output = tf.keras.layers.Reshape((self.imsize, self.imsize, self.color_channels))

    class Sampling(tf.keras.layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs, training=None):
        # Encoder branch
        x = self.flatten(inputs)
        x = self.dense_enc(x)
        z_mean = self.z_mean_layer(x)
        z_log_var = self.z_log_var_layer(x)
        z = self.sampling((z_mean, z_log_var))

        # Decoder branch
        x_dec = self.dense_dec(z)
        x_dec = self.dense_output(x_dec)
        outputs = self.reshape_output(x_dec)

        if training:
            # Add KL loss regularization during training
            kl_loss = -0.5 * tf.reduce_mean(
                z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
            )
            self.add_loss(kl_loss)

        return outputs

def my_model_function():
    # Model assumes input shape of (None, 28, 28, 1)
    model = MyModel(imsize=28, color_channels=1, intermediate_dim=64, latent_dim=32)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.MeanSquaredError()
    )
    return model

def GetInput():
    # Return a batch of 10 inputs with shape (10, 28, 28, 1) matching expected model input
    return tf.random.uniform(shape=(10, 28, 28, 1), dtype=tf.float32)


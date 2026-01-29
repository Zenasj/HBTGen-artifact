# tf.random.normal((B, 784)) ← The input to the encoder is a batch of flattened MNIST images with shape (batch_size, 784)
import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        original_dim = 784
        intermediate_dim = 64
        latent_dim = 32
        
        # Encoder network layers
        self.encoder_dense = layers.Dense(intermediate_dim, activation='relu')
        self.z_mean_layer = layers.Dense(latent_dim, name='z_mean')
        self.z_log_var_layer = layers.Dense(latent_dim, name='z_log_var')
        
        # Decoder network layers
        self.decoder_dense = layers.Dense(intermediate_dim, activation='relu')
        self.decoder_output = layers.Dense(original_dim, activation='sigmoid')
        
    def sampling(self, inputs):
        # Reparameterization trick: sample from N(z_mean, exp(z_log_var))
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def encode(self, x):
        x = self.encoder_dense(x)
        z_mean = self.z_mean_layer(x)
        z_log_var = self.z_log_var_layer(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z
    
    def decode(self, z):
        x = self.decoder_dense(z)
        return self.decoder_output(x)
    
    def call(self, inputs, training=False):
        # Forward pass: encode, sample and decode
        z_mean, z_log_var, z = self.encode(inputs)
        reconstruction = self.decode(z)
        if training:
            # Add KL divergence as a loss if training
            kl_loss = -0.5 * tf.reduce_mean(
                z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
            )
            self.add_loss(kl_loss)
        return reconstruction

def my_model_function():
    # Return an instance of the VAE model (MyModel)
    return MyModel()

def GetInput():
    # Return random input tensor matching shape (batch_size, 784) and type float32
    # Use batch size 64 as default (matching training batch size in original example)
    batch_size = 64
    original_dim = 784
    # Uniform or normal distribution okay; original training inputs were normalized MNIST digits
    return tf.random.uniform((batch_size, original_dim), minval=0., maxval=1., dtype=tf.float32)


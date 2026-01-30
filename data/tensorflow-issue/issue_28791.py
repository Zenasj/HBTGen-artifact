import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

# This model is based heavily on VAE example from Keras
# https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K
from keras.losses import mse

class VAE:
    def __init__(self, original_dim, intermediate_dim, latent_dim):
        """Creates a variational autoencoder for continuous values
        """
        self.original_dim = original_dim
        input_shape = (original_dim,)
        
        # VAE model = encoder + decoder
        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        x = Dense(intermediate_dim, activation='relu')(x)
        self.z_mean = Dense(latent_dim, name='z_mean')(x)
        self.z_log_var = Dense(latent_dim, name='z_log_var')(x)
        
        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(VAE.sampling, output_shape=(latent_dim,), name='z')([self.z_mean, self.z_log_var])
        
        # instantiate encoder model
        self.encoder = Model(inputs, [self.z_mean, self.z_log_var, z], name='encoder')
        
        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        x = Dense(intermediate_dim, activation='relu')(x)
        outputs = Dense(original_dim, activation='sigmoid')(x)

        # instantiate decoder model
        self.decoder = Model(latent_inputs, outputs, name='decoder')

        # instantiate VAE model
        outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae_mlp')
                         
    def describe(self):
        """Display model summaries and saves the architectures to PNG"""
        self.encoder.summary()
        plot_model(self.encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
        self.decoder.summary()
        plot_model(self.decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
        self.vae.summary()
        plot_model(self.vae, to_file='vae_mlp.png', show_shapes=True)
    
    def fit(self, X, optimizer='adam', **kwargs):
        """Fits the model"""
        
        def vae_loss_func(x, x_true):
            reconstruction_loss = mse(x, x_true)
            reconstruction_loss *= self.original_dim
            kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            return K.mean(reconstruction_loss + kl_loss)
        
        self.vae.compile(optimizer=optimizer, loss=vae_loss_func)
        return self.vae.fit(X, X, **kwargs)
    
    def evaluate(self, X, **kwargs):
        """Evaluate the model"""
        return self.vae.evaluate(x=X, y=X, **kwargs)
        
    # reparameterization trick
    # instead of sampling from Q(z|X), sample epsilon = N(0,I)
    # z = z_mean + sqrt(var) * epsilon
    def sampling(args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.
        # Arguments
            args (tensor): mean and log of variance of Q(z|X)
        # Returns
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]

        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

# RandomStandardNormal is not part of TensorFlow lite, so we need to use SELECT_TF_OPS to include it

converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                        tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
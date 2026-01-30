import math
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

encoder = Sequential( [
                           
        Conv2D( filters = 32, kernel_size = ( 4, 4 ), strides = ( 2, 2 ), padding = "same", activation = "relu",
                input_shape = INPUT_IMAGE_SHAPE, name = "encoder_conv2d_0" ),
        BatchNormalization( name = "encoder_batchnorm_0"),
        Conv2D( filters = 64, kernel_size = ( 4, 4 ), strides = ( 2, 2 ), padding = "same", activation = "relu", name = "encoder_conv2d_1" ),
        BatchNormalization(  name = "encoder_batchnorm_1" ),
        Flatten( name = "encoder_flatten"),
        Dense( tfpl.MultivariateNormalTriL.params_size( latent_dim ), name = "encoder_dense" ),
        tfpl.MultivariateNormalTriL( latent_dim, activity_regularizer = kl_regularizer, dtype = tf.float64, name = "encoder_outdist" )


    ], name = "encoder" )

decoder = Sequential( [
        
        Dense( units = 9 * 9 * 64, activation = "relu", input_shape = ( latent_dim, ), name = "decoder_dense" ),
        Reshape( ( 9, 9, 64 ) , name = "decoder.reshape" ),
        UpSampling2D( size = ( 2, 2 ), name = "decoder_upsampl_0" ),      # ### 18 x 18 x 32
        Conv2D( filters = 128, kernel_size = ( 3, 3 ), padding = "same", activation = "relu", name = "decoder_conv2D_0" ),
        UpSampling2D( size = ( 2, 2 ), name = "decoder_upsampl_1" ),  # ### 36 x 36 x 16
        Conv2D( filters = 128, kernel_size = ( 3, 3 ), padding = "same", activation = "relu", name = "decoder_conv2D_1" ),
        Conv2D( filters = 3, kernel_size = ( 3, 3 ), padding = "same", name = "decoder_conv2D_2" ),
        Flatten( name = "decoder_flatten" ),
        tfpl.IndependentBernoulli( event_shape = INPUT_IMAGE_SHAPE, name = "decoder_outdist" )
        
    ], name = "decoder" )

@tf.function
def reconstruction_loss(batch_of_images, decoding_dist):
    """
    This function should compute and return the average expected reconstruction loss,
    as defined above.
    The function takes batch_of_images (Tensor containing a batch of input images to
    the encoder) and decoding_dist (output distribution of decoder after passing the 
    image batch through the encoder and decoder) as arguments.
    The function should return the scalar average expected reconstruction loss.
    """
    batch_loss = decoding_dist.log_prob( batch_of_images )
    loss = - tf.math.reduce_sum( batch_loss ) / batch_of_images.shape[ 0 ]
    return loss

vae = Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs[ 0 ]) )

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
vae.compile(optimizer=optimizer, loss=reconstruction_loss)

history = vae.fit( x = train_dataset, validation_data = test_dataset, epochs = 50, verbose = True )

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers

import numpy as np

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (Dense, Flatten, Reshape, Concatenate, Conv2D, 
                                     UpSampling2D, BatchNormalization)

print( "tf.version", tf.__version__, "tfp.version", tfp.__version__ )

prior_distribution = tfd.MultivariateNormalDiag( loc = np.zeros( ( 2, ) ), scale_diag = np.ones( ( 2, ) ) )

kl_regularizer = tfpl.KLDivergenceRegularizer(
    distribution_b = prior_distribution,
    use_exact_kl = False,
    test_points_fn = lambda t : t.sample( 3 ),
    test_points_reduce_axis = None,
    weight = 1.0
    )

def reconstruction_loss(batch_of_images, decoding_dist):
    return - tf.reduce_mean( decoding_dist.log_prob( batch_of_images ) )

image_array = np.random.uniform( 0, 1, ( 100, 36, 36, 3 ) ).astype( np.float64 ) / 255.0
train_dataset = image_array[ 0 : 80 ]
test_dataset = image_array[ 80 : 100 ]

encoder = tf.keras.Sequential( [
    Conv2D( 3, 3, padding = "same", input_shape = ( 36, 36, 3 ) ),
    Flatten( input_shape = ( 36, 36, 3 ) ),
    Dense( tfpl.MultivariateNormalTriL.params_size( 2 ), name = "encoder_dense" ),
    tfpl.MultivariateNormalTriL( 2, activity_regularizer = kl_regularizer, dtype = tf.float64, name = "encoder_outdist" )
])

decoder = tf.keras.Sequential( [
    Dense( 36 * 36 * 3, input_shape = ( 2, ), name = "decoder_dense" ),
    Reshape( ( 36, 36, 3 ), name = "decoder_reshape" ),
    Conv2D( 3, 3, padding = "same", name = "decoder_conv2d" ),
    Flatten( name = "decoder_flatten" ),
    tfpl.IndependentBernoulli( event_shape = ( 36, 36, 3 ), convert_to_tensor_fn = tfd.Bernoulli.logits )
])

vae = tf.keras.Model( inputs=encoder.inputs, outputs=decoder( encoder.outputs ) )
vae.compile( optimizer = "adam", loss = reconstruction_loss )

print( "vae.summary()", vae.summary() )

vae.fit( train_dataset, epochs = 5, batch_size = 20 )

train_dataset = tf.data.Dataset.from_tensor_slices( ( image_array[ 0 : 80 ], image_array[ 0 : 80 ] ) )
test_dataset = tf.data.Dataset.from_tensor_slices( ( image_array[ 80 : 100 ], image_array[ 80 : 100 ] ) )
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

def Loss( y0, y ):
    return K.mean( K.maximum( K.abs( y0 / ( K.abs( y ) + 0.1 )), K.abs( y / ( K.abs( y0 ) + 0.1 ))))

InputLayer = tf.keras.layers.Input( dtype = 'float32', shape = ( 4, ))
Output = tf.keras.layers.Dense( 1, use_bias = True )( InputLayer )
model = tf.keras.models.Model( inputs = [InputLayer], outputs=[Output] )
model.compile( loss = Loss, optimizer = tf.keras.optimizers.Adam())
XData = np.random.random(( 100, 4 ))
YData = np.random.random(( 100, ))
model.fit( [XData], [YData], epochs = 5, batch_size = 5, verbose=2 )
model.save( 'mo.test' )
tf.keras.models.load_model( 'mo.test' )

model.save( 'mo.test' )
#each of the follow lines raises error
new_model =  tf.keras.models.load_model( 'mo.test', custom_objects = { 'Loss': Loss },compile=False)
new_model.compile( loss = Loss, optimizer = tf.keras.optimizers.Adam())
#tf.keras.models.load_model( 'mo.test' )
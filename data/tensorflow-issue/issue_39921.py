from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
import tensorflow.keras.backend as K
import tensorflow.keras as keras
class MyConcat(keras.layers.Layer):
    def __init__(self):
        super(MyConcat, self).__init__()
    def call(self, inputs):
        x, emb = inputs
        return Concatenate(axis=1)([x, emb])
    
    def compute_output_shape(self, input_shape):
        shape = (None, input_shape[0][1] + input_shape[1][1],
                input_shape[0][2], input_shape[0][3])
        return shape

inputs = Input(shape=(5,1,14))
emb = Input(shape=(512,6,18))

upconv1 = Conv2DTranspose(filters=16, 
                          kernel_size=(6,5), 
                          strides=(6,1),
                          data_format="channels_first")(inputs)

x = MyConcat()([upconv1, emb])

decoder = Model(inputs=[inputs, emb], outputs=x)      
decoder.save("decoder.tf", save_format="tf")
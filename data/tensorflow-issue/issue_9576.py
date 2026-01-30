import math
import random
from tensorflow import keras
from tensorflow.keras import layers

import os
import pdb
import traceback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import tensorflow as tf
if len(tf.config.list_physical_devices('GPU')):
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
else:
    print (' - No GPU present!! Exiting ...')
    import sys; sys.exit(1)

class Conv3DWS(tf.keras.layers.Conv3D):
    """
    Ref
     - https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv3D
     - https://www.tensorflow.org/api_docs/python/tf/nn/conv3d
     - https://github.com/joe-siyuan-qiao/WeightStandardization
    """

    def __init__(self, filters, kernel_size=(3,3,3), strides=(1, 1, 1), padding='same'
                    , dilation_rate=(1,1,1)
                    , activation='relu'
                    , kernel_regularizer=None
                    , name=''):
        super(Conv3DWS, self).__init__(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding
                        , dilation_rate=dilation_rate
                        , activation=activation
                        , kernel_regularizer=kernel_regularizer
                        , name=name)
    
    def call(self,x):

        # Step 1 - WS
        kernel_mean = tf.math.reduce_mean(self.kernel, axis=[0,1,2,3], keepdims=True, name='kernel_mean')
        kernel_std  = tf.math.reduce_std(self.kernel, axis=[0,1,2,3], keepdims=True, name='kernel_std')
        kernel_new  = (self.kernel - kernel_mean)/(kernel_std + tf.keras.backend.epsilon())
        
        # Step 2 - Convolution
        # [Does not works due to padding=same]
        output = tf.nn.conv3d(input=x, filters=kernel_new, strides=list((1,) + self.strides + (1,)), padding=self.padding.upper(), dilations=(1,) + self.dilation_rate + (1,)) 
        # [Works due to padding=valid]
        # output = tf.nn.conv3d(input=x, filters=kernel_new, strides=list((1,) + self.strides + (1,)), padding='VALID', dilations=(1,) + self.dilation_rate + (1,)) 
        
        # Step 3 - Bias
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias, data_format=self._tf_data_format)
        
        # Step 4 - Activation and return
        if self.activation is not None:
            return self.activation(output)
        else:
            return output

if __name__ == "__main__":
    try:
        x = tf.random.normal((1,140,140,40,1))
        layers = Conv3DWS(filters=10, dilation_rate=(3,3,3))
        y = layers(x)
        print (' - y: ', y.shape)

    except:
        traceback.print_exc()
        pdb.set_trace()
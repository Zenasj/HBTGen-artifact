from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
 
inputs = Input(shape=(1,), name='input_layer')
outputs = tf.identity(inputs, name='test_layer')
model = Model(inputs, outputs)
 
print(model.output_names)

['tf.identity']

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
 
inputs = Input(shape=(1,), name='input_layer')
outputs = tf.identity(inputs, name='test_layer')
model = Model(inputs, outputs)
 
print(model.output_names)

['tf_op_layer_test_layer']

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
 
inputs = Input(shape=(1,), name='input_layer')
outputs = Dense(1, name='test_layer')(inputs)
model = Model(inputs, outputs)
 
print(model.output_names)

['test_layer']

import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
 
inputs = Input(shape=(1,), name='input_layer')
x = tf.identity(inputs)
outputs = Lambda(lambda x: x, name='test_layer')(x)
model = Model(inputs, outputs)
 
print(model.output_names)

['test_layer']
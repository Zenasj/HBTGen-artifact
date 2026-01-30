from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense

def create_base_model():
    # create initial model
    a = Input([10], dtype=tf.float32)
    b = Dense(4)(a)
    c = Dense(8)(b)
    model = Model(inputs={'a': a}, outputs={'b': b, 'c': c})
    return model

def extend_base_model(model):
    # create a new model to attach additional layers to tensor "c"
    c = model.outputs[-2]
    d = Dense(16)(c)
    
    # notice here, since model.inputs and model.outputs are flattened to a lists,
    # we lose all naming information
    new_model = Model(inputs=model.inputs, outputs=model.outputs+[d])
    return new_model
    
base = create_base_model()
model = extend_base_model(base)

# we lose all naming information in the inputs and outputs given by the base model
# outputs: [<tensor>, <tensor> ,  <tensor>]
model(tf.zeros([1, 10], dtype=tf.float32))

a = Input([10], dtype=tf.float32)
b = Dense(4)(a)
model = Model(inputs={'a': a}, outputs={'b': b})
type(model.inputs)  # list
type(model.outputs) # list

a = Input([10], dtype=tf.float32)
b = Dense(4)(a)
model = Model(inputs={'a': a}, outputs={'b': b})
type(model.inputs)  # dict
type(model.outputs) # dict

import tensorflow as tf
c = tf.keras.Input(2, name='c')                          
d = tf.keras.Input(2, name='d')                          
z = tf.keras.layers.Add()([c*10, d])                     
j = tf.keras.layers.Add()([z, z])                        
model = tf.keras.Model(inputs={'c':c, 'd': d}, outputs={'z': z, 'j': j})

out = model.outputs[0]  # does `out` represent `z` or `j`?
k = tf.keras.layers.add()([out, out, out])
extended_model = tf.keras.Model(inputs=model.inputs, outputs=model.outputs+[k])

import tensorflow as tf

c = tf.keras.Input(2, name='c')
d = tf.keras.Input(2, name='d')
out = tf.keras.layers.Add()([c*10, d])
model = tf.keras.Model([c, d], [out])

model({'c': tf.ones([1, 2]) * 2, 'd': tf.ones([1, 2]) * 3}).numpy() 
# As expected, this outputs: array([[23,  23]])

model({'c': tf.ones([1, 2]) * 2, 'a': tf.ones([1, 2]) * 3}).numpy()
# Unexpectedly, this outputs: array([[32, 32]])
# I would expect that this call throws an exception
from tensorflow import keras
from tensorflow.keras import layers

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# NOT NESTED
inp = tf.keras.Input((4,))
y = tf.keras.layers.Dense(4, name="od_1")(inp)
y = tf.keras.layers.Dense(2, name="od_2")(y)
y = tf.keras.layers.Dense(4, name="id_1")(y)
y = tf.keras.layers.Dense(10, name="od_3")(y)
y = tf.keras.layers.Dense(10, name="od_4")(y)
final_model = tf.keras.Model(inputs=[inp], outputs=[y])
final_model.summary()

sub_model = tf.keras.Model(inputs=[final_model.input], outputs=[final_model.get_layer("id_1").output])
sub_model.summary()

# NESTED
inp_1 = tf.keras.Input(shape=(2,))
x = tf.keras.layers.Dense(4, name="id_1")(inp_1)
inner_model = tf.keras.Model(inputs=[inp_1], outputs=[x], name="inner_model")

inp_outer = tf.keras.Input((4,))
y = tf.keras.layers.Dense(4, name="od_1")(inp_outer)
y = tf.keras.layers.Dense(2, name="od_2")(y)
y = inner_model(y)
y = tf.keras.layers.Dense(10, name="od_3")(y)
y = tf.keras.layers.Dense(10, name="od_4")(y)
final_model = tf.keras.Model(inputs=[inp_outer], outputs=[y])
final_model.summary()

sub_model = tf.keras.Model(inputs=[final_model.input], outputs=[final_model.get_layer("inner_model").get_layer("id_1").output])
sub_model.summary()

import tensorflow as tf

# Define an inner model of an input and a single dense layer
inner_input = tf.keras.layers.Input(10)
x = inner_input
x = tf.keras.layers.Dense(10, name="inner_layer")(x)
inner_output = x

inner_model = tf.keras.Model(inner_input, inner_output, name='inner_model')
inner_model.summary()

# Define an outer model in which we prepend a single Dense layer before the inner model.
# The inner model is thus a layer or sub-model within the outer model
outer_input = tf.keras.layers.Input(10)
x = outer_input
x = tf.keras.layers.Dense(10, name="outer_layer")(x)
outer_output = inner_model(x)

outer_model = tf.keras.Model(outer_input, outer_output, name='outer_model')
outer_model.summary()

# Append an extra Dense layer after `inner_layer`, and create a new model with inputs
# the `inner_input` and outputs the output of the newly appended Dense layer.
# This works.
x = inner_model.get_layer('inner_layer').output
x = tf.keras.layers.Dense(10)(x)
extended_inner_model_output = x
extended_inner_model = tf.keras.Model(inner_input, extended_inner_model_output)
extended_inner_model.summary()

# Append an extra Dense layer after `inner_layer`, and create a new model with inputs
# the `outer_input`(!) and outputs the output of the newly appended Dense layer.
# This does not work.
x = outer_model.get_layer('inner_model').get_layer('inner_layer').output
x = tf.keras.layers.Dense(10)(x)
extended_inner_model_output = x
extended_inner_model = tf.keras.Model(outer_input, extended_inner_model_output)
extended_inner_model.summary()
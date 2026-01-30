from tensorflow import keras

import tensorflow as tf

inputs = tf.keras.Input([])
var = tf.Variable(3.0)

# OK
tf.keras.Model(inputs=inputs, outputs=[inputs*var])

# AttributeError exception
tf.keras.Model(inputs=inputs, outputs=[inputs*var,var])
# The same happens with variants like `tf.identity(var)`
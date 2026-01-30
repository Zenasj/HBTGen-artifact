from tensorflow.keras import layers

import tensorflow as tf
from tensorflow import keras

# tf.config.experimental_run_functions_eagerly(True)
var = tf.Variable([[3.0]])
model = keras.Sequential([keras.layers.Dense(1, input_shape=(1,))])
model.compile(loss="mse", optimizer="adam")

tf.print(var)  # should print 3, OK
var.assign(model.inputs[0])
tf.print(var)  # should print anything else but 3, or raise an error - but prints 3
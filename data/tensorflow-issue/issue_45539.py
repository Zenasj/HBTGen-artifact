from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow.python.ops.parallel_for.gradients import batch_jacobian

# Define function that computes f(x) = x^2 and its derivative df/dx = 2*x
@tf.function
def square(x):    
    y = x**2
    dydx = batch_jacobian(y, x)
    return y, dydx

# Create a model that uses the function
x = tf.keras.backend.placeholder(shape=(None, 2), dtype=tf.float32)
y, dydx = tf.keras.layers.Lambda(lambda c: square(c))(x)
model = tf.keras.Model(x, [y, dydx])

# Test case: evaluate model with dummy input
x_input = tf.constant([[1., 2.], [3., 4.], [5., 6.]])
y_output, dydx_output = model(x_input)
print(y_output)
print(dydx_output)
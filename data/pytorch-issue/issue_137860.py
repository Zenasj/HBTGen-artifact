import math

import torch
import tensorflow as tf
import jax.numpy as jnp
import numpy as np
from keras.layers import Lambda

# Check if numpy's acos function is accessible
try:
    print(f"Numpy acos function: {np.acos}")
except AttributeError as e:
    print("Numpy acos function is not available:", e)

# Input data (as provided)
input_data = np.array([
    [
        [
            0.5834500789642334,
            0.057789839804172516
        ],
        [
            0.13608910143375397,
            0.8511932492256165
        ],
        [
            -0.8579278588294983,
            -0.8257414102554321
        ],
        [
            -0.9595631957054138,
            0.665239691734314
        ],
        [
            0.5563135147094727,
            0.7400242686271667
        ],
        [
            0.9572367072105408,
            0.5983171463012695
        ],
        [
            -0.07704125344753265,
            0.5610583424568176
        ],
        [
            -0.7634511590003967,
            0.2798420190811157
        ],
        [
            -0.7132934331893921,
            0.8893378376960754
        ]
    ]
])

# PyTorch acos operation
def torch_acos(x):
    return torch.acos(torch.tensor(x, dtype=torch.float32))

# TensorFlow acos operation
def tf_acos(x):
    return tf.acos(tf.convert_to_tensor(x, dtype=tf.float32))

# Keras acos operation
def keras_acos(x):
    return Lambda(lambda x: tf.math.acos(x))(tf.convert_to_tensor(x, dtype=tf.float32))

# JAX acos operation
def jax_acos(x):
    return jnp.arccos(jnp.array(x))

# Chainer acos operation
def chainer_acos(x):
    return np.arccos(x)

# Calculate results
pytorch_result = torch_acos(input_data)
tensorflow_result = tf_acos(input_data)
keras_result = keras_acos(input_data).numpy()  # Convert Keras result to numpy
jax_result = jax_acos(input_data)
chainer_result = chainer_acos(input_data)

# Print results
print(f"PyTorch acos result: {pytorch_result.detach().numpy()}")  # Detach to convert to numpy
print(f"TensorFlow acos result: {tensorflow_result.numpy()}")
print(f"Keras acos result: {keras_result}")
print(f"JAX acos result: {jax_result}")
print(f"Chainer acos result: {chainer_result}")

import torch

torch.manual_seed(0)
input_data = torch.randn(100)

torch.set_printoptions(precision=10, profile="full")
print(input_data.cos()[:5])
input_data[3] = 1e10
print(input_data.cos()[:5])
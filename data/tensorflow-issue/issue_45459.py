from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

# a model that simply take a float as input and return it Model(x) = x
inputs = tf.keras.Input(shape=(1,))
outputs = tf.keras.layers.Dense(
    1, 
    activation=None, 
    use_bias=False, 
    kernel_initializer=tf.keras.initializers.Ones()
)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

def get_generator(batch_size=32, max_value=100):
    for begin in np.arange(start=0, stop=max_value, step=batch_size):
        end = min(max_value, begin + batch_size)
        print("Call generator {} to {}".format(begin, end))
        yield np.arange(begin, end)[:,None]

batch_size = 32
max_value = 100
data_generator = get_generator(batch_size=batch_size, max_value=max_value)

N = 3
# 3 steps but generator called 4 times...
res = model.predict(x=data_generator, steps=N)

expected_result = np.arange(0, N * batch_size)[:, None]
print(np.allclose(res, expected_result))
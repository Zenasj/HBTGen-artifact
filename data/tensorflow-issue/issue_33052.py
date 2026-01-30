from tensorflow import keras
from tensorflow.keras import layers

python
import tensorflow as tf
import numpy as np
import timeit

use_eager = False
use_v2 = False

if not use_eager:
    tf.compat.v1.disable_eager_execution()
if not use_v2:
    tf.compat.v1.disable_control_flow_v2()


n_steps = 1000
n_input = 100
n_hidden = 1000
batch_size = 64

inputs = tf.keras.Input((n_steps, n_input))
outputs = tf.keras.layers.SimpleRNN(units=n_hidden, return_sequences=True)(inputs)
outputs = tf.keras.layers.Dense(units=n_input)(outputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=tf.optimizers.SGD(0.1), loss="mse")

x = np.ones((batch_size, n_steps, n_input))
y = np.ones((batch_size, n_steps, n_input))

# warmup
model.fit(x, y, epochs=1)

start = timeit.default_timer()
model.fit(x, y, epochs=10)
print("Execution time:", timeit.default_timer() - start)

python
import tensorflow as tf
import numpy as np
import timeit


n_steps = 1000
n_input = 100
n_hidden = 1000
batch_size = 64
with tf.device("/gpu:0"):
    inputs = tf.keras.Input((n_steps, n_input))
    outputs = tf.keras.layers.SimpleRNN(units=n_hidden, return_sequences=True)(inputs)
    outputs = tf.keras.layers.Dense(units=n_input)(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.optimizers.SGD(0.1), loss="mse")

    x = np.ones((batch_size, n_steps, n_input))
    y = np.ones((batch_size, n_steps, n_input))

    # warmup
    model.fit(x, y, epochs=1)

    start = timeit.default_timer()
    model.fit(x, y, epochs=10)
    print("Execution time (defaults):", timeit.default_timer() - start)


tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_control_flow_v2()

with tf.device("/gpu:0"):
    inputs = tf.keras.Input((n_steps, n_input))
    outputs = tf.keras.layers.SimpleRNN(units=n_hidden, return_sequences=True)(inputs)
    outputs = tf.keras.layers.Dense(units=n_input)(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.optimizers.SGD(0.1), loss="mse")

    x = np.ones((batch_size, n_steps, n_input))
    y = np.ones((batch_size, n_steps, n_input))

    # warmup
    model.fit(x, y, epochs=1)

    start = timeit.default_timer()
    model.fit(x, y, epochs=10)
    print("Execution time (no eager, no v2):", timeit.default_timer() - start)
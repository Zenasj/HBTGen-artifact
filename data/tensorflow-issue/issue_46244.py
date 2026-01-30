import math
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

# build inputs(b, t, i) == t + 1 with size [batch, timesteps, inputs]
batch, timesteps, ninputs = 1, 5, 1
timevec = (
    np.arange(timesteps, dtype=np.float32) + 1
)  # we start at one since the first gets double-counted
inputs = np.broadcast_to(
    timevec[np.newaxis, :, np.newaxis], (batch, timesteps, ninputs)
)


class AddsLossLayer(keras.layers.Layer):
    """Identity layer which calls add_loss on mean of input"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # inputs is batch x ninputs
        # return input but add_loss(mean of current input over ninputs dimension, should == timestep + 1)
        self.add_loss(tf.math.reduce_mean(inputs, axis=0, name="loss"))
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape


adds_loss_layer = AddsLossLayer()
td_layer = keras.layers.TimeDistributed(adds_loss_layer)

outputs = td_layer(inputs)

print(f"inputs shape = {inputs.shape}")
print(f"outputs shape = {outputs.shape}")
print(f"len(td_layer.losses) == {len(td_layer.losses)}")
print(f"len(adds_loss_layer.losses) == {len(adds_loss_layer.losses)}")
print("loss values == ", [loss.numpy() for loss in td_layer.losses])

# inputs shape = (3, 5, 1)
# outputs shape = (3, 5, 1)
# len(td_layer.losses) == 6
# len(adds_loss_layer.losses) == 6
# loss values ==  [array([1.], dtype=float32), array([1.], dtype=float32), array([2.], dtype=float32), array([3.], dtype=float32), array([4.], dtype=float32), array([5.], dtype=float32)]

# so we double count the first timestep!

model_inputs = tf.keras.Input(shape=inputs.shape[1:], dtype="float32")
adds_loss_layer = AddsLossLayer()
td_layer = keras.layers.TimeDistributed(adds_loss_layer, name="td")
model_outputs = td_layer(model_inputs)

model = keras.Model(model_inputs, model_outputs)
model.compile()
model.evaluate(inputs, None, verbose=2)

# 1/1 - 0s - loss: 3.0000
# (this is correct, (1 + 2 + 3 + 4 + 5) / 5 == 3)

model.fit(inputs, None, epochs=1)

# 1/1 [==============================] - 0s 100ms/step - loss: 3.0000
# (this is correct, (1 + 2 + 3 + 4 + 5) / 5 == 3)

outputs = model(inputs)

print("mean model.losses = ", tf.math.reduce_mean(model.losses).numpy())

# mean model.losses =  2.6666667
# due to double counted first timestep, (1 + 1 + 2 + 3 + 4 + 5) / 6 = 2.666667
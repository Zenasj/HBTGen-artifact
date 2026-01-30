import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np

from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Layer
from tensorflow.keras import Input, Model

tf.compat.v1.disable_eager_execution()
# tf.config.run_functions_eagerly(False)


def loss_function(inputs):
    return tf.keras.backend.switch(tf.size(input=inputs) > 0, tf.keras.backend.mean(inputs), tf.constant(0.0))


class RpnClassLoss(Layer):
    def __init__(self, name="loss_function", **kwargs):
        super(RpnClassLoss, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        loss = loss_function(inputs)
        return loss


class Output(Layer):
    def __init__(self, **kwargs):
        super(Output, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs


inputs = Input(shape=(3,))
output = Output()(inputs)
# loss = tf.convert_to_tensor(loss_function(inputs))
loss = RpnClassLoss()(inputs)
outputs = [output, loss]
model = Model(inputs, outputs)

loss = tf.reduce_mean(loss, keepdims=True)
model.add_loss(loss)

model.summary()

model.compile(optimizer="adam", loss=[None] * len(model.outputs))
model.fit(np.random.random((2, 3)))

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
print("Try model_from_json")
loaded_model = model_from_json(loaded_model_json, custom_objects={"Output": Output, "RpnClassLoss": RpnClassLoss})  # ValueError: Inconsistent values for attr 'Tidx' DT_FLOAT vs. DT_INT32 while building NodeDef 'tf_op_layer_Mean_1/Mean_1'
print("model_from_json ok")

loss = tf.reduce_mean(loss, keepdims=True)
model.add_loss(loss)

class RpnClassLoss(Layer):
    def call(self, inputs):
        loss = loss_function(inputs)
        loss = tf.reduce_mean(loss, keepdims=True)
        self.add_loss(loss)
        return loss

import tensorflow as tf
import numpy as np

from tensorflow.keras.models import model_from_json
from tensorflow.keras import Input, Model

tf.compat.v1.disable_eager_execution()

inputs = Input(shape=(3,))
output = inputs * 2
output_loss = tf.keras.backend.mean(inputs)
outputs = [output, output_loss]
model = Model(inputs, outputs)

loss = tf.reduce_mean((output_loss * 0.9))
model.add_loss(loss)

model.summary()

model.compile(optimizer="adam", loss=[None] * len(model.outputs))
model.fit(np.random.random((2, 3)))

model_json = model.to_json()
json_filename = "model.json"
with open(json_filename, "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

json_file = open(json_filename, 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)  # ValueError: Inconsistent values for attr 'Tidx' DT_FLOAT vs. DT_INT32 while building NodeDef 'tf_op_layer_Mean_1/Mean_1'
print("model_from_json ok")
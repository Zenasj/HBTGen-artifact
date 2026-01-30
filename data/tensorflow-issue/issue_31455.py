from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
tf.enable_eager_execution()
print(tf.__version__)
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
import numpy as np

in0 = Input(shape=(1,), dtype="float32", name='my_input_0')
in1 = Input(shape=(1,), dtype="float32", name='my_input_1')
concatted = Lambda(lambda inputs: tf.concat(inputs, axis=-1))([in0,in1])

outputs = Dense(3)(concatted)

#------------- way 1 (does not work)-----------------

# outs = [Lambda(lambda outputs: outputs[...,i], name=f'output_{i}')(outputs) for i in range(3)]

#------------- way 2 (does not work)-----------------

outs = []
for i in range(3):
  outs.append(Lambda(lambda outputs: outputs[...,i], name=f'output_{i}')(outputs))

#------------- way 3 (does work)-----------------

# out0 = Lambda(lambda outputs: outputs[...,0], name='my_output_0')(post_process)
# out1 = Lambda(lambda outputs: outputs[...,1], name='my_output_1')(post_process)
# out2 = Lambda(lambda outputs: outputs[...,2], name='my_output_2')(post_process)
# outs=[out0, out1, out2]

#----------------------------

my_model = Model(inputs=[in0,in1], outputs=outs)
tf.keras.backend.learning_phase = 0

my_model.predict([np.array([[.5],[.3]]), np.array([[-.1],[.2]])])

tf.saved_model.save(my_model, './mymodel')
reloaded = tf.saved_model.load_v2('./mymodel')
sig = reloaded.signatures['serving_default']
sig(my_input_0=tf.constant(np.array([[.5],[.3]]), dtype=tf.float32), my_input_1=tf.constant(np.array([[-.1],[.2]]), dtype=tf.float32))
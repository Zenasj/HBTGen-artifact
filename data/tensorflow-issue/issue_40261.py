import random
from tensorflow.keras import layers

from tensorflow.keras import Input, Model
import time
import numpy as np

x = Input(shape=(1, 1))
model = Model(inputs=x, outputs=x)

t = time.time()
i = 0
while i<100:
    model.predict(np.zeros((1, 1, 1)))
    i += 1
print(time.time() - t)

x = Input(shape=(1, 1))
model = Model(inputs=x, outputs=x)

t = time.time()
i = 0
while i<100:
    model.predict(np.zeros((1, 1, 1)))
    i += 1
print(time.time() - t)

x = Input(shape=(1, 1))
model = Model(inputs=x, outputs=x)

t = time.time()
i = 0
while i<100:
    model(np.zeros((1, 1, 1)), training=False)
    i += 1
print(time.time() - t)

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import GRU, Dense
import time
import numpy as np

x = Input(shape=(15, 60))

rnn = GRU(50, dropout=0.5, recurrent_dropout=0.5)(x)

y = Dense(1)(rnn)

model = Model(inputs=x, outputs=y)

t = time.time()
i = 0
while i<100:
    model.predict(np.random.rand(1, 15, 60))
    i += 1
print(time.time() - t)

@tf.function
def serve(x):
  return model(x, training=False)

while i<100:
    train_dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(1, 15, 60)).batch(1)
    model.predict(train_dataset)
    i += 1

model.call = tf.function(model.call, experimental_relax_shapes)
model(..., training=False)

self.model.predict(state) # slow
self.model(state, training=False) # fast

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.disable_v2_behavior()

sess = tf.Session()
with sess.as_default():

    G_A2B = model_from_json(loded_model_json, custom_objects={'ReflectionPadding3D':ReflectionPadding3D, 'InstanceNormalization':InstanceNormalization})
    G_A2B.load_weights(model_path+'G_A2B_model_epoch_'+str(model_num)+'.hdf5')
    place_holder = tf.compat.v1.placeholder(tf.float32, shape=(1,170,110,110,1))
    G_A2B = G_A2B(place_holder, training=False)

def AI_simulation(AI_input):

    result = sess.run(G_A2B, feed_dict={place_holder: AI_input[np.newaxis, :, :, :, np.newaxis]})
    result = np.squeeze(result)

    return result

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.disable_v2_behavior()

sess = tf.Session()
with sess.as_default():

    G_A2B = model_from_json(loded_model_json, custom_objects={'ReflectionPadding3D':ReflectionPadding3D, 'InstanceNormalization':InstanceNormalization})
    G_A2B.load_weights(model_path+'G_A2B_model_epoch_'+str(model_num)+'.hdf5')


def AI_simulation(AI_input):

    result = G_A2B(AI_input[np.newaxis, :, :, :, np.newaxis])
    result = sess.run(result )
    result = np.squeeze(result)

    return result

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.disable_v2_behavior()

G_A2B = model_from_json(loded_model_json, custom_objects={'ReflectionPadding3D':ReflectionPadding3D, 'InstanceNormalization':InstanceNormalization})
G_A2B.load_weights(model_path+'G_A2B_model_epoch_'+str(model_num)+'.hdf5')
G_A2B.call = tf.function(G_A2B.call, experimental_relax_shapes = True)

def AI_simulation(AI_input):

    result = G_A2B.predict(AI_input[np.newaxis, :, :, :, np.newaxis])
    result = np.squeeze(result)
    return result

import tensorflow as tf  #tf2

G_A2B = model_from_json(loded_model_json, custom_objects={'ReflectionPadding3D':ReflectionPadding3D, 'InstanceNormalization':InstanceNormalization})
G_A2B.load_weights(model_path+'G_A2B_model_epoch_'+str(model_num)+'.hdf5')
G_A2B.call = tf.function(G_A2B.call, experimental_relax_shapes = True)

def AI_simulation(AI_input):
    result = G_A2B(AI_input[np.newaxis, :, :, :, np.newaxis])
    result = np.squeeze(result.numpy())
    return result
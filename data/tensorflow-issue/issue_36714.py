import math
import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np
import requests
import json

x1_ = tf.keras.Input(shape=(64,256,))
x2_ = tf.keras.Input(shape=(512,))


W1 = tf.keras.layers.Dense(512)
W2 = tf.keras.layers.Dense(512)

x1 = W1(x1_)
x2 = tf.expand_dims(x2_, 1)
x2 = W2(x2)
y = tf.math.add(x1, x2)

Model = tf.keras.Model([x1_, x2_], [y])
model = Model

tf.saved_model.save(model, "/tmp/models/model/1/")

x1 = np.random.normal(size=(1,64,256))
x2 = np.zeros((1, 512), dtype=np.float32)

values = [x1.tolist(), x2.tolist()]
inputs = {t.name[:-2]:t for t in model.inputs}

d = dict(zip(inputs, values))
data = {"instances": [d]}
data = json.dumps(data)

r = requests.post('http://localhost:8504/v1/models/tensorflow_model:predict', data=data)
print(r.content.decode('utf-8'))
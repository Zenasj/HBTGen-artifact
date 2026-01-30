import random
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import Sequence

#%% Create dummy Sequential
model = Sequential([Input((224, 224, 3)), GlobalAveragePooling2D(), Dense(1)])
model.summary()
assert model.built
model.input_shape
tf.saved_model.save(model, "model")

#%% Make prediction with loaded model and Sequence
tf_model = load_model("model")
class DataGen(Sequence):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __getitem__(self, index):
        return self.data[index * self.batch_size : (index + 1) * self.batch_size]

    def __len__(self):
        return len(self.data) // self.batch_size


X = np.random.rand(16, 224, 224, 3)
tf_model.predict(DataGen(X, batch_size=2))
# ValueError: Please provide model inputs as a list or tuple of 2 or 3 elements: (input, target) or (input, target, sample_weights) Received tf.Tensor(...)

#%% Manual build does not fix the issue
assert not tf_model.built
tf_model.build((None, 224, 224, 3))
assert tf_model.built
tf_model.predict(DataGen(X, batch_size=2))

#%% Try with tf.data.Dataset instead: OK
tf_model.predict(tf.data.Dataset.from_tensor_slices(X).batch(2))

#%% Call first on np.array does fix the issue
tf_model.predict(X[:2])
tf_model.predict(DataGen(X, batch_size=2))

#%% But model still does not have input_shape and has "multiple" in summary
tf_model.inputs
tf_model.input_shape
tf_model.summary()
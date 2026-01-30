import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf

def to_complex(x):
    return tf.dtypes.complex(x, x)
    
base_model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False)
x = base_model.output
x = tf.keras.layers.Lambda(to_complex)(x)


model = tf.keras.Model(inputs=base_model.input, outputs=x)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
X = np.random.rand(10000, 224, 224, 3).astype("float32")
out = model.predict(X, batch_size=64)
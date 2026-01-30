from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import pickle
import tensorflow as tf
print(tf.version.GIT_VERSION, tf.version.VERSION, flush=True)

model_input = tf.keras.Input(shape=(1,), dtype=tf.int64)
lookup = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=['a', 'b'])(model_input)
output = tf.keras.layers.Dense(10)(lookup)
full_model = tf.keras.Model(model_input, output)

# this part works
try:
    model_bytes = pickle.dumps(full_model)
    model_recovered = pickle.loads(model_bytes)
except Exception as e:
    print("Failed! Error:", e, flush=True)
else:
    print("Success!", flush=True)

# this part throws an error
try:
    full_model.save("/tmp/temp_model")
    full_model_loaded = tf.keras.models.load_model("/tmp/temp_model")
    model_bytes = pickle.dumps(full_model_loaded)
    model_recovered = pickle.loads(model_bytes)
except Exception as e:
    print("Failed! Error:", e, flush=True)
else:
    print("Success!", flush=True)
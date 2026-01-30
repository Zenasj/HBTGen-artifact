import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models

# this works
def build_model():
    ...
    return tf.keras.Model()

model1 = tf.keras.models.load_model("path/to/checkpoint.tf")

model2 = build_model()
model2.load_weights("path/to/weights.tf")

model3 = build_model()
model3.load_weights("path/to/checkpoint.tf")

# this works
new_model = build_model()
new_model.load_weights("path/to/checkpoint.tf/variables/variables")

def build_model():
    ...

path = "path/to/checkpoint.tf"
callbacks = [tf.keras.callbacks.ModelCheckpoint(path, save_weights_only=False)]
model = build_model()
model.fit(..., callbacks=callbacks)

# now I can either load the whole model
new_model = tf.keras.models.load_model(path)

# or I can only load the weights
another_new_model = build_model()
another_new_model = another_new_model.load_weights(path)
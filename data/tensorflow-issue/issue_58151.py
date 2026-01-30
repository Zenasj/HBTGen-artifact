from tensorflow import keras
from tensorflow.keras import models

# Model works with an input, then breaks with the same input after you save and load it
# Must be related to the broadcasting happening with the language input (as it is wrongly set to (None,) instead of ())

import tensorflow as tf


class InnerModel(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()
        # Optionally some layers here

    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        audio, lang = inputs
        # Optionally some processing here
        return lang


def get_model():
    audio = tf.keras.Input(shape=(None, 1))
    lang = tf.keras.Input(shape=(None,), dtype=tf.string)  # Fixed by setting shape=()
    inner_model = InnerModel()
    out = inner_model([audio, lang])
    return tf.keras.Model(inputs=[audio, lang], outputs=[out])


model = get_model()
_ = model([tf.zeros((1, 1000, 1)), tf.constant(["fr"])], training=False)

# Save + load
model.save("test")  # Also breaks with tf.keras.models.save_model(model, "test")
model = tf.keras.models.load_model("test")

# Do inference again
_ = model([tf.zeros((1, 1000, 1)), tf.constant(["fr"])], training=False)
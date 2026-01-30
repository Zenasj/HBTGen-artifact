from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

@tf.keras.saving.register_keras_serializable(package="MyPackage")
class DummyModel(tf.keras.Model):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.sublayer = tf.keras.layers.Dense(16)
    def call(self, x, **kw):
        a = x["a"]
        b = x["b"]
        c = x["c"]
        return self.sublayer(tf.concat([a,b,c], axis=-1))

model = DummyModel()
out = model(dict(
    a = tf.keras.Input(3,dtype=tf.float32),
    b = tf.keras.Input(4,dtype=tf.float32),
    c = tf.keras.Input(5,dtype=tf.float32),
    )
)
model.summary()

model.save("temp.keras")

tf.keras.saving.load_model("temp.keras")
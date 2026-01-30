import math
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
print(f"TensorFlow version: {tf.version.VERSION}")

def get_implements_signature():
    implements_signature = [
        'name: "exp_sin"',
        'attr {key: "tfl_fusable_op" value { b: true } }',
    ]
    return " ".join(implements_signature)


@tf.function(experimental_implements=get_implements_signature())
def exp_sin(x):
    x = tf.math.exp(x)
    x = tf.math.sin(x)
    return x


class custom_example(tf.keras.layers.Layer):
    def call(self, inputs):
        return exp_sin(inputs)


x_in = x = tf.keras.layers.Input(shape=(10, 10, 3), batch_size=1)
x = custom_example()(x)
x = tf.keras.layers.Conv2D(filters=8, kernel_size=4)(x)
model = tf.keras.Model(inputs=x_in, outputs=x, name="example")
model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
with open("model_fp32.tflite", "wb") as f:
    f.write(converter.convert())
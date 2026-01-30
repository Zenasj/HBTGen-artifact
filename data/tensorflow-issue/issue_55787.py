from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf


class Layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return {"out1": tf.repeat(1, batch_size), "out2": tf.repeat(2, batch_size)}


def build_model():

    input = tf.keras.layers.Input(shape=(1,))
    out = Layer(name="my_layer")(input)

    return tf.keras.Model(
        inputs=input,
        outputs=out,
    )


model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss={"out1": "mse"},
    metrics={"out2": "mse"},
)

data = (
    # x
    [5, 5, 5, 5, 5, 5, 5, 5],
    # y
    {
        "out1": [5, 5, 5, 5, 5, 5, 5, 5],
        "out2": [5, 5, 5, 5, 5, 5, 5, 5],
    },
)


ds = tf.data.Dataset.from_tensor_slices(data)
ds = ds.batch(2)

model.fit(ds)
from tensorflow import keras

import tensorflow as tf

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(10),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1),
    ]
)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    out = model(tf.zeros((1, 10)), training=True)

import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():


  model = tf.keras.Sequential([
        tf.keras.layers.Dense(10),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1),
    ])

  def forward():
    return model(tf.zeros((1, 10)), training=True)
    
  print(strategy.experimental_run_v2(forward, args=()))

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(10),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1),
    ]
)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    out = model(tf.zeros((1, 10)), training=True)
    print(out)

import tensorflow as tf
from tensorflow.keras import layers

class Model(tf.Module):

    def __init__(self):
        super().__init__()

        with self.name_scope:
            self.layers = [
                layers.Conv2D(10, (3, 3)),
                layers.BatchNormalization()
            ]

    def no_param_call(self, input):
        x = input
        for layer in self.layers:
            x = layer(x)

        return x


    def param_call(self, input):
        x = input
        for layer in self.layers:
            x = layer(x, training=True)

        return x


def app():
    strategy = tf.distribute.MirroredStrategy()
    image_dimension = []
    with strategy.scope():
        model = Model()
        no_param_forward_fn = tf.function(model.no_param_call, autograph=False).get_concrete_function(
                tf.TensorSpec([1, 64, 64, 3], tf.float32)
        )
        print('no param call succeeded')
        param_forward_fn = tf.function(model.param_call, autograph=False).get_concrete_function(
            tf.TensorSpec([1, 64, 64, 3], tf.float32)
        )
        print('param call succeeded')


if __name__ == '__main__':
    app()

my_model = Model(...)
tf.keras.Model(inputs=[tf.keras.layers.Input(shape=...)], outputs=my_model.call(training=True))
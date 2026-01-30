from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

class TestLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(TestLayer, self).__init__(**kwargs)

        self.static_layer = tf.keras.layers.Dense(128)
        self.my_layers = tf.python.training.tracking.data_structures.List()
        for i in range(4):
            layer = tf.keras.layers.Dense(128)
            self.my_layers.append(layer)


    def call(self, x):
        x = self.static_layer(x)

        for layer in self.my_layers:
            x = layer(x)

        return x


    def get_config(self):
        return super(TestLayer, self).get_config()


model = tf.keras.Sequential([TestLayer()])

x = tf.constant(42.0, shape=[1,1])
y1 = model(x)

model.save('my_test_model', save_format='tf')

model_loaded = tf.keras.models.load_model('my_test_model')
y2 = model_loaded(x)

# output
model.summary()
model_loaded.summary()
print('n vars: ', len(model.weights), ' ', len(model_loaded.weights))
print('diff: ', tf.norm(y1-y2))
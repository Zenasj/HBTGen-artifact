from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow import keras


class LayerWithSublayers(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = keras.layers.Embedding(3, 2)
        self.dense = keras.layers.Dense(4)

    def call(self, inputs, **kwargs):
        return self.dense(self.embedding(inputs))


input_ids = keras.Input(shape=(4,), dtype=tf.int32, name='input_ids')
output_layer = LayerWithSublayers()
output_layer.embedding.trainable = False
output = output_layer(input_ids)

model = keras.Model(inputs=[input_ids], outputs=[output])
model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model_file_name = 'foo.h5'
model.save(filepath=model_file_name)

loaded_model = keras.models.load_model(model_file_name, custom_objects={'LayerWithSublayers': LayerWithSublayers})

import tensorflow as tf
from tensorflow import keras


class LayerWithSublayers(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = keras.layers.Embedding(3, 2)
        self.dense = keras.layers.Dense(4)

    def call(self, inputs, **kwargs):
        return self.dense(self.embedding(inputs))


input_ids = keras.Input(shape=(4,), dtype=tf.int32, name='input_ids')
output_layer = LayerWithSublayers()
output_layer.embedding.trainable = False
output = output_layer(input_ids)

model = keras.Model(inputs=[input_ids], outputs=[output])
model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model_file_name = 'foo.tf'

model.save(filepath=model_file_name)

loaded_model = keras.models.load_model(model_file_name, custom_objects={'LayerWithSublayers': LayerWithSublayers})
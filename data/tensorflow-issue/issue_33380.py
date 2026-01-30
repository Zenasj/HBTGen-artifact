from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

attention = tf.keras.layers.Attention()
attention.get_config()

import tensorflow as tf

attention_kwargs = dict(use_scale=True, name='attention')
attention = tf.keras.layers.Attention(**attention_kwargs)
attention.get_config = lambda: attention_kwargs  # manually make a get_config()

# Build and save the model
query = tf.keras.layers.Input((None, 10))
value = tf.keras.layers.Input((None, 10))
outputs = attention([query, value])
model = tf.keras.Model(inputs=[query, value], outputs=outputs)
model.save('model.h5', save_format='h5')

# Restore the model
restored_model = tf.keras.models.load_model(
    filepath='model.h5',
    custom_objects={'Attention': tf.keras.layers.Attention},
)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

base_model = tf.keras.applications.ResNet50(
    include_top=False, weights='imagenet',
    input_shape=[96,112,3])

x = base_model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, use_bias=True, name='Bottleneck')(x)
   
model = tf.keras.models.Model( base_model.input, x )

tf.compat.v1.lite.converter.from_frozen_graph()

import tensorflow as tf
tf.__version__
# 2.3.2

graph_def_file = "resface36_L2Norm128_at_epoch_168.pb"
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(graph_def_file,
                                              ["input"],
                                              ["resface36/Bottleneck/BiasAdd"],
                                              input_shapes={'input': [1, 96, 112, 3]})
tflite_model = converter.convert()

open("resface36_L2Norm128_at_epoch_168.tflite", "wb").write(tflite_model)
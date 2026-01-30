import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

3
import numpy as np
import tensorflow as tf

# Generate tf.keras model.
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=(1, 9), strides=(1, 1), padding='same', data_format='channels_last'))
# model.add(tf.keras.layers.Dense(2, input_shape=(3,)))
model.compile(loss=tf.keras.losses.MSE,
              optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              metrics=[tf.keras.metrics.categorical_accuracy],
              sample_weight_mode='temporal')


x = np.random.random((1, 1, 9, 64))
y = np.random.random((1, 1, 9, 64))
model.train_on_batch(x, y)
model.predict(x)

# Save tf.keras model in H5 format.
keras_file = 'keras_model.h5'
tf.keras.models.save_model(model, keras_file)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)

def representative_dataset_gen():
    for _ in range(100):
        yield [np.ones([1,1,9,64]).astype(np.float32)]
# converter.quantized_input_stats = {'input' : (0., 1.)}
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
# converter.inference_type = tf.uint8
converter.representative_dataset = representative_dataset_gen

tflite_model = converter.convert()

# Save the model.
with open('model2.tflite', 'wb') as f:
  f.write(tflite_model)
import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2DTranspose


num_filters=512
inputs=tf.keras.Input((32,32,1024))

outputs= Conv2DTranspose(filters=num_filters,
                    kernel_size=(2, 2), 
                    strides=2, 
                    padding="same",name='convTrans')(inputs)

base_model=tf.keras.Model(inputs=inputs,outputs=outputs,name="test_transConv2D_model")
base_model.summary()
print(base_model.layers[1].weights[0].shape)
print(base_model.layers[1].weights[1].shape)
# (2, 2, 512, 1024)
# (512,)

import tensorflow_model_optimization as tfmot
quant_aware_model = tfmot.quantization.keras.quantize_model(base_model)

x_train = np.random.randn(4,32, 32, 1024).astype(np.float32)
y_train = np.random.randn(4, 64, 64, 512).astype(np.float32)
quant_aware_model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy']
)
quant_aware_model.fit(x_train, y_train,epochs=1)
quant_aware_model.summary()

quant_aware_model.input.set_shape((1,) + quant_aware_model.input.shape[1:])

converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

quantized_tflite_model = converter.convert()
tflite_model_filename='test_transConv2D_model_uint8_inout.tflite'
with open(tflite_model_filename, 'wb') as f:
    f.write(quantized_tflite_model)
    print("wirte tflite file done!")
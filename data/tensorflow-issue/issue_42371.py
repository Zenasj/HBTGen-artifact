import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TFConvTranspose1d(tf.keras.Model):

    def __init__(self, channels, ksize, stride, padding):
        super(TFConvTranspose1d, self).__init__()
        self.conv1d_transpose = tf.keras.layers.Conv2DTranspose(
            filters = channels,
            kernel_size = (ksize, 1),
            strides = (stride, 1),
            padding = "same",
        )

    def call(self, x):
#         x = tf.expand_dims(x, axis=2)
        
#         print(" CONV TRANSPOSE 1D ", x.shape)
        x = self.conv1d_transpose(x)
#         print(" CONV TRANSPOSE 1D ", x.shape)
#         x = tf.squeeze(x, axis=2)
        return x

checkModel = TFConvTranspose1d(256, 16, 8, "same")
input_shape = (1, 100,1,  512)
checkModel.build(input_shape)

x = np.random.rand(1, 100,1,  512)
x = x.astype('float32')

out = checkModel.predict(x)
print( "KERAS OUTPUT : ", out.shape)

print(" Number of Params : ", checkModel.count_params())

tflite_converter = tf.lite.TFLiteConverter.from_keras_model(checkModel)
tflite_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_converter.allow_custom_ops = True
tflite_converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_converter.post_training_quantize = True
tfmodel = tflite_converter.convert()

open("convTranspose.tflite", "wb").write(tfmodel)
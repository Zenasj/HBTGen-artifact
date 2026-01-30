import tensorflow as tf
from tensorflow import keras

class C3BR(tf.keras.Model):

			def __init__(self, filterNum, kSize, strSize, padMode):
				super(C3BR, self).__init__()
				self.conv = layers.Conv3D(filters=filterNum, kernel_size=kSize, strides=strSize, padding=padMode, data_format='channels_first')
				self.BN = layers.BatchNormalization(axis=1)
			
			def call(self, inputs):
				x = self.conv(inputs)
				x= self.BN(x)
				return activations.relu(x)
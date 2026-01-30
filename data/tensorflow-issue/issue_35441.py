import random
from tensorflow import keras
from tensorflow.keras import optimizers

from tensorflow.keras.mixed_precision import experimental as mixed_precision
curOpt = tf.keras.optimizers.Adam(learning_rate=lr)
curOpt = mixed_precision.LossScaleOptimizer(curOpt, loss_scale='dynamic')
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
# query if it has been well set
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)
from MedNN import SimpleUNet
TUNet = SimpleUNet(modelInDim, classNum)
TUNet.build_model(input_shape=(mbSize,)+modelInDim)

x0 =tf.random.uniform((mbSize, 4, 64, 64, 64), dtype=tf.float32)
y=TUNet(x0, True)
TUNet.summary()
TUNet._set_inputs(x0, True)
# Method #1
tf.saved_model.save(TUNet, r'...\TUNet_Test')
# Method #2
TUNet.save(r'...\TUNet_Test', save_format='tf')

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, activations

class C3BR(tf.keras.Model):
		''' 3D Convolution + Batch Normalisation + Relu '''
		def __init__(self, filterNum, kSize, strSize, padMode):
			super(C3BR, self).__init__()
			self.conv = layers.Conv3D(filters=filterNum, kernel_size=kSize, strides=strSize, padding=padMode, data_format='channels_first')
			self.BN = layers.BatchNormalization(axis=1)
		
		def call(self, inputs, ifTrain=True):
			x = self.conv(inputs)
			if ifTrain == True:
				x= self.BN(x)
			return activations.relu(x)

		def build_model(self, input_shape):
			''' A work-around to define dimensions of signals through the NN'''
			self.build(input_shape)
			inputs = tf.keras.Input(shape=input_shape[1:])
			_ = self.call(inputs) 

class SimpleUNet1(tf.keras.Model):
	"""
	Serialise basic units so as to build up a double-layered encoder-decoder U-Net
	Input:
		inDim: (for initialisation) [modaility/channel, tensor dimensions]
		classNum: background included
		name: name for the net
		inputs: 5D tf tensor of [mbSize, modaility/channel, tensor dimensions]. Inputs must be organised into channel first order
		input_shape: a 1X5 tuple [mbSize, modaility/channel, tensor dimensions]
		ifTrain: True for training, and False for validation and testing
	Returns:
		outputs: 5D tf tensor of [mbSize, classNum, tensor dimensions]
	"""
	def __init__(self, inDim, classNum, name='SimpleUNet', **kwarg):
		super(SimpleUNet1, self).__init__(name=name, **kwarg)
		self.inDim = inDim
		self.classNum = classNum
		dimEnSt1End = np.array(inDim)[1:]-2-2
		dimEnSt2Ed = dimEnSt1End/2-2-2
		dimBridgeEnd = (dimEnSt2Ed/2-2-2)*2
		dimDEStd1End = (dimBridgeEnd-2-2)*2
		self.outDim = dimDEStd1End-2-2-2
		temp = ((dimEnSt2Ed - dimBridgeEnd)/2).astype('int32')
		crop3d1 = tuple(np.tile(temp, (2, 1)).T)
		temp = ((dimEnSt1End - dimDEStd1End)/2).astype('int32')
		crop3d2 = tuple(np.tile(temp, (2, 1)).T)

		self.en_st1_cbr1 = C3BR(32, 3, 1, 'valid')
		self.en_st1_cbr2 = C3BR(64, 3, 1, 'valid')
		self.en_st2_mp = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', data_format='channels_first')
		self.en_st2_cbr1 = C3BR(64, 3, 1, 'valid')
		self.en_st2_cbr2 = C3BR(128, 3, 1, 'valid')
		self.bridge_mp = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', data_format='channels_first')
		self.bridge_cbr1 = C3BR(128, 3, 1, 'valid')
		self.bridge_cbr2 = C3BR(256, 3, 1, 'valid')    
		self.bridge_tconv1 = layers.Conv3DTranspose(256, 2, strides=2, padding='valid', data_format='channels_first')
		self.de_3dcrop1 = layers.Cropping3D(crop3d1, data_format='channels_first')
		self.de_st1_concat = layers.Concatenate(axis=1)
		self.de_st1_cbr1 = C3BR(256, 3, 1, 'valid')
		self.de_st1_cbr2 = C3BR(128, 3, 1, 'valid')    
		self.de_st1_tconv1 = layers.Conv3DTranspose(128, 2, strides=2, padding='valid', data_format='channels_first')
		self.de_3dcrop2 = layers.Cropping3D(crop3d2, data_format='channels_first')
		self.de_st2_concat = layers.Concatenate(axis=1)
		self.de_st2_cbr1 = C3BR(64, 3, 1, 'valid')
		self.de_st2_cbr2 = C3BR(64, 3, 1, 'valid') 
		self.final_conv3D = layers.Conv3D(filters=self.classNum, kernel_size=3, strides=1, padding='valid', data_format='channels_first')
				
	#@tf.function
	def call(self, inputs, ifTrain=True):
		x0 = self.en_st1_cbr1(inputs, ifTrain)
		xEnSt1End = self.en_st1_cbr2(x0, ifTrain)
		x1 = self.en_st2_mp(xEnSt1End)
		x2 = self.en_st2_cbr1(x1, ifTrain)
		xEnSt2Ed = self.en_st2_cbr2(x2, ifTrain)
		x3 = self.bridge_mp(xEnSt2Ed)  
		x4 = self.bridge_cbr1(x3, ifTrain)
		x5 = self.bridge_cbr2(x4, ifTrain)    
		xBridgeEnd = self.bridge_tconv1(x5)
		xCrop1 = self.de_3dcrop1(xEnSt2Ed)
		print(xBridgeEnd.shape)
		print(xCrop1.shape)
		x6 = self.de_st1_concat([xBridgeEnd, xCrop1])
		print(x6.shape)
		x7 = self.de_st1_cbr1(x6, ifTrain)
		x8 = self.de_st1_cbr2(x7, ifTrain)
		xDeSt1End = self.de_st1_tconv1(x8)
		xCrop2 = self.de_3dcrop2(xEnSt1End)
		x9 = self.de_st2_concat([xDeSt1End, xCrop2])
		x10 = self.de_st2_cbr1(x9, ifTrain)
		x11 = self.de_st2_cbr2(x10, ifTrain)
		x12 = self.final_conv3D(x11)
		outputs = activations.softmax(x12, axis=1)
		
		return outputs
		
	def build_model(self, input_shape):
		''' A work-around to define dimensions of signals through the NN'''
		self.build(input_shape)
		inputs = tf.keras.Input(shape=input_shape[1:])

		_ = self.call(inputs)
		
	def compute_output_shape(self):
		# Override this function if one expects to use the subclassed model in Kera's fit() method; Otherwise, it is optional.
		return tf.TensorShape(np.append(self.classNum, self.outDim))    

modelInDim = (4, 64, 64, 64)
classNum = 2
mbSize = 2
TUNet = SimpleUNet1(modelInDim, classNum)
TUNet.build_model(input_shape=(mbSize,)+modelInDim)
TUNet.summary()
TUNet.save(r'e:\TTweight', save_format='tf')

x6 = layers.concatenate([xBridgeEnd, xCrop1], axis=1)
x9 = layers.concatenate([xDeSt1End, xCrop2], axis=1)

x=tf.random.uniform((mbSize, 4, 64, 64, 64))
y=TUNet(x)
# use your directory in lieu of r'...\TTweight'
TUNet.save(r'...\TTweight', save_format='tf')

x=tf.random.uniform((mbSize, 4, 64, 64, 64))
TUNet._set_inputs(x)
TUNet.save(r'...\TTweight', save_format='tf')

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, activations

class C3BR(tf.keras.Model):
  ''' 3D Convolution + Batch Normalisation + Relu '''
  def __init__(self, filterNum, kSize, strSize, padMode):
    super(C3BR, self).__init__()
    self.conv = layers.Conv3D(filters=filterNum, kernel_size=kSize, strides=strSize, padding=padMode, data_format='channels_first')
    self.BN = layers.BatchNormalization(axis=1)

  def call(self, inputs, ifTrain=True):
    x = self.conv(inputs)
    if ifTrain == True:
      x= self.BN(x)
    return activations.relu(x)

  def build_model(self, input_shape):
    ''' A work-around to define dimensions of signals through the NN'''
    self.build(input_shape)
    inputs = tf.keras.Input(shape=input_shape[1:])
    _ = self.call(inputs)

class SimpleUNet1(tf.keras.Model):
  """
  Serialise basic units so as to build up a double-layered encoder-decoder U-Net
  Input:
    inDim: (for initialisation) [modaility/channel, tensor dimensions]
    classNum: background included
    name: name for the net
    inputs: 5D tf tensor of [mbSize, modaility/channel, tensor dimensions]. Inputs must be organised into channel first order
    input_shape: a 1X5 tuple [mbSize, modaility/channel, tensor dimensions]
    ifTrain: True for training, and False for validation and testing
  Returns:
    outputs: 5D tf tensor of [mbSize, classNum, tensor dimensions]
  """
  def __init__(self, inDim, classNum, name='SimpleUNet', **kwarg):
    super(SimpleUNet1, self).__init__(name=name, **kwarg)
    self.inDim = inDim
    self.classNum = classNum
    dimEnSt1End = np.array(inDim)[1:]-2-2
    dimEnSt2Ed = dimEnSt1End/2-2-2
    dimBridgeEnd = (dimEnSt2Ed/2-2-2)*2
    dimDEStd1End = (dimBridgeEnd-2-2)*2
    self.outDim = dimDEStd1End-2-2-2
    temp = ((dimEnSt2Ed - dimBridgeEnd)/2).astype('int32')
    crop3d1 = tuple(np.tile(temp, (2, 1)).T)
    temp = ((dimEnSt1End - dimDEStd1End)/2).astype('int32')
    crop3d2 = tuple(np.tile(temp, (2, 1)).T)

    self.en_st1_cbr1 = C3BR(32, 3, 1, 'valid')
    self.en_st1_cbr2 = C3BR(64, 3, 1, 'valid')
    self.en_st2_mp = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', data_format='channels_first')
    self.en_st2_cbr1 = C3BR(64, 3, 1, 'valid')
    self.en_st2_cbr2 = C3BR(128, 3, 1, 'valid')
    self.bridge_mp = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', data_format='channels_first')
    self.bridge_cbr1 = C3BR(128, 3, 1, 'valid')
    self.bridge_cbr2 = C3BR(256, 3, 1, 'valid')
    self.bridge_tconv1 = layers.Conv3DTranspose(256, 2, strides=2, padding='valid', data_format='channels_first')
    self.de_3dcrop1 = layers.Cropping3D(crop3d1, data_format='channels_first')
    self.de_st1_concat = layers.Concatenate(axis=1)
    self.de_st1_cbr1 = C3BR(256, 3, 1, 'valid')
    self.de_st1_cbr2 = C3BR(128, 3, 1, 'valid')
    self.de_st1_tconv1 = layers.Conv3DTranspose(128, 2, strides=2, padding='valid', data_format='channels_first')
    self.de_3dcrop2 = layers.Cropping3D(crop3d2, data_format='channels_first')
    self.de_st2_concat = layers.Concatenate(axis=1)
    self.de_st2_cbr1 = C3BR(64, 3, 1, 'valid')
    self.de_st2_cbr2 = C3BR(64, 3, 1, 'valid')
    self.final_conv3D = layers.Conv3D(filters=self.classNum, kernel_size=3, strides=1, padding='valid', data_format='channels_first')

  #@tf.function
  def call(self, inputs, ifTrain=True):
    x0 = self.en_st1_cbr1(inputs, ifTrain)
    xEnSt1End = self.en_st1_cbr2(x0, ifTrain)
    x1 = self.en_st2_mp(xEnSt1End)
    x2 = self.en_st2_cbr1(x1, ifTrain)
    xEnSt2Ed = self.en_st2_cbr2(x2, ifTrain)
    x3 = self.bridge_mp(xEnSt2Ed)
    x4 = self.bridge_cbr1(x3, ifTrain)
    x5 = self.bridge_cbr2(x4, ifTrain)
    xBridgeEnd = self.bridge_tconv1(x5)
    xCrop1 = self.de_3dcrop1(xEnSt2Ed)
    print(xBridgeEnd.shape)
    print(xCrop1.shape)
    x6 = layers.concatenate([xBridgeEnd, xCrop1], axis=1)
    print(x6.shape)
    x7 = self.de_st1_cbr1(x6, ifTrain)
    x8 = self.de_st1_cbr2(x7, ifTrain)
    xDeSt1End = self.de_st1_tconv1(x8)
    xCrop2 = self.de_3dcrop2(xEnSt1End)
    x9 = layers.concatenate([xDeSt1End, xCrop2], axis=1)
    x10 = self.de_st2_cbr1(x9, ifTrain)
    x11 = self.de_st2_cbr2(x10, ifTrain)
    x12 = self.final_conv3D(x11)
    outputs = activations.softmax(x12, axis=1)

    return outputs

  def build_model(self, input_shape):
    ''' A work-around to define dimensions of signals through the NN'''
    self.build(input_shape)
    inputs = tf.keras.Input(shape=input_shape[1:])

    _ = self.call(inputs)

  def compute_output_shape(self):
    # Override this function if one expects to use the subclassed model in Kera's fit() method; Otherwise, it is optional.
    return tf.TensorShape(np.append(self.classNum, self.outDim))

modelInDim = (4, 64, 64, 64)
classNum = 2
mbSize = 2
TUNet = SimpleUNet1(modelInDim, classNum)
TUNet.build_model(input_shape=(mbSize,)+modelInDim)
# TUNet.summary()

x=tf.random.uniform((mbSize, 4, 64, 64, 64))
TUNet._set_inputs(x)
# use your directory in lieu of r'...\TTweight'
TUNet.save(r'TTweight', save_format='tf')
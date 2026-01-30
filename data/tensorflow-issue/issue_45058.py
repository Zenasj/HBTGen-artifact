import tensorflow as tf
from tensorflow import keras

def vgg_layers(layer_names):
	vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
	outputs = [vgg.get_layer(name).output for name in layer_names]
	model = tf.keras.Model([vgg.input], outputs)
	return model

class VGGloss(Model):
	def __init__(self):
		super(VGGloss, self).__init__()
		layers = [f'block{i+1}_conv1' for i in range(5)]
		self.layerweights = [1./32, 1./16, 1./8, 1./4, 1.]
		self.vgg = vgg_layers(layers)
		self.vgg.trainable = False
		self.l1_loss = tf.keras.losses.MeanAbsoluteError('auto')

	def call(self, x, y):
		x_vgg, y_vgg = self.vgg(x), self.vgg(y), 
		loss = 0
		for w, xi, yi in zip(self.layerweights, x_vgg, y_vgg):
			loss += w*self.l1_loss(xi, yi)
		return loss
import random
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import Sequence

class CustomCallback(Callback):
	def __init__(self, patience=10, restore_best=True, dir="weights"):
		super().__init__()
		self.best_score = 0.0
		self.best_epoch = 0
		self.best_weights = None
		self.patience = patience
		self.restore_best = restore_best

	def on_epoch_end(self, epoch, logs={}):
		score = logs["val_acc"]
		if score>self.best_score:
			self.best_score = score
			self.best_epoch = epoch
			self.best_weights = self.model.get_weights()
			print(f"\nBest Score: {self.best_score*100:.2f}")
		else: print(f"\nBest Epoch: {self.best_epoch+1}, Best Score: {self.best_score*100:.2f}")

		if self.patience>0 and epoch-self.best_epoch >= self.patience:
			print(f"\nStopping... Best Epoch: {self.best_epoch+1}, Best Score: {self.best_score*100:.2f}")
			self.stopped_epoch = epoch
			self.model.stop_training = True
			if self.restore_best: 		self.model.set_weights(self.best_weights)
    
class DataGenerator(Sequence):
	def __init__(self,x, y, batch_size=32):
		self.x = x
		self.y = y
		self.batch_size = batch_size
		self.on_epoch_end()

	def __len__(self):
		return int(np.ceil(len(self.x) / float(self.batch_size)))

	def on_epoch_end(self):
		self.indexes = np.arange(len(self.x))
		np.random.shuffle(self.indexes)
	
	def __getitem__(self, index):
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
		x_batch = self.x[indexes]
		y_batch = self.y[indexes]
		return x_batch, y_batch

def dense_model(input_shape, output_shape):
	inputs = Input(input_shape)
	x = Flatten()(inputs)
	x = Dense(300, activation='relu')(x)
	x = Dense(100, activation="relu")(x)
	outputs = Dense(output_shape, activation='softmax')(x)
	model = Model(inputs=inputs, outputs=outputs)
	model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
	return model

(x_train, y_train), (x_test, y_test) = mnist.load_data()
height, width = 28, 28
input_shape = (height, width)
output_shape = 10
x_train = x_train/255
x_test = x_test/255
y_train = to_categorical(y_train, output_shape)
y_test = to_categorical(y_test, output_shape)
model = dense_model(input_shape, output_shape)
history = model.fit_generator(DataGenerator(x_train, y_train), epochs=12, validation_data=DataGenerator(x_test, y_test), verbose=0)
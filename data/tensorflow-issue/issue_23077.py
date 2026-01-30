import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

tf.contrib.distribute.MirroredStrategy(['/device:CPU:0', '/device:CPU:1', '/device:GPU:0', '/device:GPU:1', '/device:GPU:2', '/device:GPU:3'])

distribution = tf.contrib.distribute.MirroredStrategy()

def buildModel(word_index, embeddings_index, nClasses, 
MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, gpusno, nLayers=3,Number_Node=100, dropout=0.5):
	
	embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
			
	model = tf.keras.models.Sequential()
	
	model.add(tf.keras.layers.Embedding(len(word_index) + 1,
								EMBEDDING_DIM,
								weights=[embedding_matrix],
								input_length=MAX_SEQUENCE_LENGTH,
								trainable=True))
	model.add(tf.keras.layers.Flatten())
	
	for i in range(0,nLayers):
		model.add(tf.keras.layers.Dense(Number_Node, activation='relu'))
		model.add(tf.keras.layers.Dropout(dropout))
	
	model.add(tf.keras.layers.Dense(nClasses, activation='softmax'))

	
	distribution = tf.contrib.distribute.MirroredStrategy()
        # i cannot add argument num_gpus, as tf will say it cannot find the devices.
	
	model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
				  optimizer=tf.train.AdamOptimizer(),
				  metrics=['accuracy'], 
				  distribute=distribution)
				  
	print('model summary:') 
	model.summary()

	return model
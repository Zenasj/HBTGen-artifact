from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

second_half  = Model(inputs=model.get_layer('dense_alpha').input,
                            outputs=model.get_layer('dense_out').output)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

inp = Input((300,480,3))
base_model = Xception(include_top=False,
                              weights='imagenet',
                              input_tensor=inp,
                              pooling='avg')

for layer in base_model.layers[:65]:
	layer.trainable = False
for layer in base_model.layers[65:]:
	layer.trainable = True
	
	
x = base_model(inp)

x = Dropout(0.5, name='base_drop')(x)
dense_alpha = Dense(1024, activation='relu', name='dense_alpha')(x)
dense_alpha = Dropout(0.5, name='dense_alpha_drop')(dense_alpha)
dense_alpha = Dense(2, activation='softmax', name='dense_out')(dense_alpha)

losses = [tf.keras.losses.BinaryCrossentropy()]

model = Model(inputs=inp, outputs=[dense_alpha])
model.compile(optimizer=Adam(lr=0.0001),
			  loss=losses,
			  metrics=['accuracy'])

model_base = Model(inputs=model.input, outputs=model.get_layer('base_drop').output)
                
second_part = K.function([model.get_layer('dense_alpha').input, K.learning_phase()],
                            [model.get_layer('dense_out').output])

second_part = Model([model.get_layer('dense_alpha').input],
                            [model.get_layer('dense_out').output])

second_part = Model([model.get_layer('dense_alpha').input],
                            [model.get_layer('dense_out').output])
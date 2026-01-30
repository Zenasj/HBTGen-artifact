from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np
print('TensorFlow', tf.__version__)

'''
Case 1: Simple model without any regularization
'''
input_layer = tf.keras.Input(shape=[10])
x = tf.keras.layers.Dense(units=16, activation='relu')(input_layer)
x = tf.keras.layers.Dense(units=16, activation='relu')(x)
output_layer = tf.keras.layers.Dense(units=4, activation=None)(x)
model_a = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])

assert model_a.losses == []


'''
Case 2: Simple model with regularization added
        during layer creation
'''
input_layer = tf.keras.Input(shape=[10])
x = tf.keras.layers.Dense(units=16, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(l=1e-5))(input_layer)
x = tf.keras.layers.Dense(units=16, activation='relu', 
                          kernel_regularizer=tf.keras.regularizers.l2(l=1e-5))(x)
output_layer = tf.keras.layers.Dense(units=4, activation=None, 
                                     kernel_regularizer=tf.keras.regularizers.l2(l=1e-5))(x)
model_b = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])

assert model_b.losses != []


'''
Case 3: For a prebuilt model, for example, keras_applications models
        and other models if they are already built, manually adding regularization
        to layers does not show up in model.losses
'''
input_layer = tf.keras.Input(shape=[10])
x = tf.keras.layers.Dense(units=16, activation='relu')(input_layer)
x = tf.keras.layers.Dense(units=16, activation='relu')(x)
output_layer = tf.keras.layers.Dense(units=4, activation=None)(x)
model_c = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])

for layer in model_c.layers:
    if hasattr(layer, 'kernel_regularizer'):
        setattr(layer, 'kernel_regularizer', tf.keras.regularizers.l2(l=1e-5))
        
for layer in model_c.layers:
    if hasattr(layer, 'kernel_regularizer'):
        assert getattr(layer, 'kernel_regularizer').l2 == np.array([1e-5], dtype='float32')
        
assert model_c.losses == []

model = tf.keras.applications.ResNet50()
for layer in model.layers:
    if hasattr(layer, 'kernel_regularizer'):
        setattr(layer, 'kernel_regularizer', tf.keras.regularizers.l2(l=1e-5))
        
for layer in model.layers:
    if hasattr(layer, 'kernel_regularizer'):
        assert getattr(layer, 'kernel_regularizer').l2 == np.array([1e-5], dtype='float32')
        
assert model.losses == []
model.save_weights('resnet50.h5', save_format='h5')

new_model = tf.keras.models.model_from_json(model.to_json())
# new_model would have random weights
new_model.load_weights('resnet50.h5')
assert new_model.losses != [] #works fine
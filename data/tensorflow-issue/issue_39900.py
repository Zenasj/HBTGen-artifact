from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, load_model


layer_input = Input(shape=(None, 10), name='input')
layer_dense = Dense(12, activity_regularizer=regularizers.l2(), name='dense')(layer_input)
layer_lambda = Lambda(lambda batch: batch, activity_regularizer=regularizers.l2(), name='lambda')(layer_dense)

model = Model(inputs=layer_input, outputs=layer_lambda)
model.compile(loss='mean_squared_error')

print('* Original model *')
print('regularizer losses', model.losses)
print('regularizer dense', model.get_layer('dense').activity_regularizer)
print('regularizer lambda', model.get_layer('lambda').activity_regularizer)
print()

model.save('test.h5')
model_reloaded = load_model('test.h5')

print('* Reloaded model *')
print('regularizer losses', model_reloaded.losses)
print('regularizer dense', model_reloaded.get_layer('dense').activity_regularizer)
print('regularizer lambda', model_reloaded.get_layer('lambda').activity_regularizer)
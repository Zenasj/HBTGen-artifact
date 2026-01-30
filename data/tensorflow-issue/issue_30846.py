from tensorflow.keras import layers
from tensorflow.keras import models

from keras.layers import Input, Dense, Dot, Reshape, Flatten
from keras.models import Model

model_in = Input((10,), name='input')
model_out = model_in
model_out = Reshape((-1, 1), name='reshape')(model_out)
model_out = Dot(axes=2, name='dot')([model_out, model_out])
model_out = Flatten(name='flatten')(model_out)
model = Model(model_in, model_out)
model.summary()
model.compile()
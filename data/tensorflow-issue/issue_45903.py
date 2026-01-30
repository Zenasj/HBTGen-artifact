import random
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam

def cmetrics(y_true, y_pred):
	return(0)

model = Sequential()
model.add(Dense(10,activation="relu", input_shape=(331, 331, 3)))
model.add(Flatten())
model.add(Dense(10, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
	optimizer=Adam(),
	metrics=[cmetrics])
model.summary()
xdata = np.random.rand(100,331,331,3)
ydata = np.random.rand(100,10)
history = model.fit(x=xdata, y=ydata)
model.save('test.h5', save_format='h5')
model = load_model('test.h5', custom_objects={'cmetrics': cmetrics,})
history = model.fit(x=xdata,y=ydata)

model = load_model("mymodel_best.h5", custom_objects={"mymetric":mymetric}, compile=True)
model.compile(loss=model.loss, optimizer=model.optimizer, metrics=[mymetric])
import random
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

4.520166488224531 
4.409448146820068 
4.139582633972168

4.520166488224531 
4.520166488224531 
4.520166488224531

from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
import numpy as np
x = np.random.normal(1, 2, 10)
y = np.random.normal(1, 2, 10)

model = Sequential()
model.add(Input(1))
model.add(Dense(10))
model.add(Activation("sigmoid"))
model.add(Dense(1))
model.summary()


rmsprop = RMSprop(lr=0.01)
model.compile(loss='mean_squared_error', optimizer=rmsprop)
hist = model.fit(x=x,y=y)
_ = model.evaluate(x, y, verbose=0)
pred = model.predict(x)
print(np.mean((y-pred)**2), '\n', hist.history['loss'][0], '\n', _)
import random
from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend as K
import numpy as np

ipt = Input((16,))
x   = Dense(16)(ipt)
out = Dense(16)(x)
model = Model(ipt, out)
model.compile('sgd', 'mse')

outs_fn = K.function([model.input, K.symbolic_learning_phase()],
                     [model.layers[1].output])  # error
x = np.random.randn(32, 16)
print(outs_fn([x, True]))
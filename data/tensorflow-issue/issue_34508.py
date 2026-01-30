import random
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np

ipt = Input((16,))
out = Dense(16)(ipt)
model = Model(ipt, out)
model.compile('adam', 'mse')

x = np.random.randn(32, 16)
model.train_on_batch(x, x)

grads = model.optimizer.get_gradients(model.total_loss, model.layers[-1].output)
grads_fn = K.function(inputs=[model.inputs[0], model._feed_targets[0], model.sample_weights[0]], 
                      outputs=grads)

import tensorflow.python.keras.backend as K
learning_phase = K.symbolic_learning_phase()
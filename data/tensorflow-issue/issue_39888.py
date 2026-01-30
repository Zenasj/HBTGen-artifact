import random
from tensorflow.keras import layers
from tensorflow.keras import models

print(sample_weight)
try: print("sample_weight =", K.eval(sample_weight))
except: pass
print(loss, '\n')

# EAGER
Tensor("ExpandDims:0", shape=(32, 1), dtype=float32)
Tensor("mean_squared_error/weighted_loss/value:0", shape=(), dtype=float32)

Tensor("ExpandDims:0", shape=(32, 1), dtype=float32)
Tensor("mean_squared_error/weighted_loss/value:0", shape=(), dtype=float32)

# GRAPH
1.0
sample_weight = 1.0
Tensor("loss/conv2d_loss/weighted_loss/Mul:0", shape=(32, 28, 28), dtype=float32)

Tensor("conv2d_sample_weights:0", shape=(None,), dtype=float32)
sample_weight = [1.]
Tensor("loss_1/conv2d_loss/weighted_loss/Mul:0", shape=(32, 28, 28), dtype=float32)

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
tf.compat.v1.disable_eager_execution()

batch_shape = (32, 28, 28, 1)

ipt = Input(batch_shape=batch_shape)
out = Conv2D(filters=1, kernel_size=(1, 1))(ipt)
model = Model(ipt, out)
model.compile('adam', 'mse')

x = y = np.random.randn(*batch_shape)
sw = np.ones(len(x))

model.train_on_batch(x, y, sw)

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras.models import Model

tf.compat.v1.disable_eager_execution()

batch_shape = (32, 28, 28, 1)

ipt = Input(batch_shape=batch_shape)
x   = Conv2D(filters=1, kernel_size=(1, 1))(ipt)
x   = Flatten()(x)
out = Dense(10, activation='softmax')(x)
model = Model(ipt, out)
model.compile('adam', 'categorical_crossentropy')

x = np.random.randn(*batch_shape)
n_classes, batch_size = 10, 32
class_labels = np.random.randint(0, n_classes, batch_size)
y = np.eye(n_classes)[class_labels]
sw = np.random.uniform(0, 2, (len(x),))

model.train_on_batch(x, y, sw)
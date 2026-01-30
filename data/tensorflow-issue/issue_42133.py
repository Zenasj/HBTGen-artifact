import random
from tensorflow.keras import layers
from tensorflow.keras import models

import shutil
import tempfile
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard

#%%############################################
ipt = Input(shape=(16,))
out = Dense(16)(ipt)
model = Model(ipt, out)
model.compile('adam', 'mse')

logdir = tempfile.mkdtemp()
print('tensorboard --logdir="%s"' % logdir)
tb = TensorBoard(logdir, write_graph=True)

#%%############################################
x = y = np.random.randn(160, 16)
model.fit(x, y, batch_size=32, callbacks=[tb])

# shutil.rmtree(logdir)
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

tf.compt.v1.disable_eager_execution()

x = tf.keras.Input(shape=(3,))
y = tf.keras.layers.Dense(2)(x)
model = tf.keras.Model([x], [y])
compile_kwargs = {'optimizer': 'sgd', 'loss': tf.keras.losses.mean_squared_error}
model.compile(**compile_kwargs)

data = [np.full(shape=(3,), fill_value=i) for i in range(10)]
data = np.asarray(data)
labels = [np.full(shape=(2,), fill_value=i)*3 for i in range(10)]
labels = np.asarray(labels)

ds = tf.data.Dataset.from_tensor_slices((data, labels)).batch(2)

# Option 1 - doesn't work when eager execution is disabled
# Throws - RuntimeError: __iter__() is only supported inside of tf.function or when eager execution is enabled.
# ins = ds.map(lambda x, y: x)
# for batch in ins:
   # r = model.test_on_batch(batch)
   # print(r)

# Option 2
@tf.function
def test_on_batch_example(model, ins):
  for batch in ins:
    r = model.test_on_batch(batch)
    print(r)
  
ins = ds.map(lambda x, y: x)
test_on_batch_example(model, ins)
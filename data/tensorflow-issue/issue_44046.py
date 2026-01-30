from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

# make a source dataset
source_data=tf.data.Dataset.range(100)
# take two windows of different size onto the source dataset as
# our inputs and outputs
# and batch them up into 20 data points
inputs=source_data.window(10,1,1,True).batch(20,True)
targets=source_data.window(5,1,1,True).batch(20,True)
# zip this into a single training data set
training_set=tf.data.Dataset.zip((inputs,targets))
# a very simple network that takes input of [batch_size,10] and outputs [batch_size,5]
inLayer=tf.keras.layers.Input(shape=(10,),batch_size=20)
outLayer=tf.keras.layers.Dense(5)(inLayer)
model=tf.keras.Model(inputs=inLayer,outputs=outLayer)
model.compile(optimizer='adam',loss='mse')
# this fit call fails
model.fit(training_set)

d=tf.data.Dataset.range(10)
w=d.window(3,1,1)
w = w.flat_map(lambda x:x).batch(3).batch(2) #flat map but do nothing to values, then batch twice to get what model.fit expects from windowed datasets

keys=('a', 'b', 'c')
window_size = 2
ds = tf.data.Dataset.from_tensor_slices(
    {'a': [1, 2, 3],
     'b': [4, 5, 6],
     'c': [7, 8, 9]})

# map dict to Dataset
ds = ds.flat_map(lambda features_dict: tf.data.Dataset.zip(tuple([tf.data.Dataset.from_tensors(v) for v in features_dict.values()])))
# get properly windowed Dataset
ds = ds.window(window_size, shift=1, drop_remainder=True)
ds = ds.flat_map(lambda *features_ds: tf.data.Dataset.zip(tuple([f.batch(window_size) for f in features_ds])))
# map back to dict
ds = ds.map(lambda *features_ds_tuple: {key: f for key, f in zip(keys, features_ds_tuple)})

for v in ds:
    print(f"iteration: ",  v)
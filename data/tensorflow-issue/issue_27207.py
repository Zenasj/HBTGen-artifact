import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).shuffle(buffer_size=10000).batch(10)
# single node
ds_strategy=tf.distribute.MirroredStrategy()
# cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
# multinode
# ds_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
with ds_strategy.scope():
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    model.compile(loss='mse', optimizer='sgd')
    model.fit(dataset,epochs=200)
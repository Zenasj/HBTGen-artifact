from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.reshape((10000,28,28,1))
y_test = tf.keras.utils.to_categorical(y=y_test)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(8, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation='softmax'))
dataset = tf.data.Dataset.from_tensor_slices((x_test)) # I need to provide y_test also until version 1.12.0
dataset = dataset.batch(batch_size=10)
data = dataset.make_one_shot_iterator()
output = model.predict(x=data,steps=1000,verbose=True)

train_iterator = train_data.make_initializable_iterator()
valid_iterator = valid_data.make_initializable_iterator()

# initialize tables and iterators using sess from tf.keras
init_sess = tf.keras.backend.get_session()
init_sess.run(train_iterator.initializer)
init_sess.run(valid_iterator.initializer)
init_sess.run(tf.tables_initializer())
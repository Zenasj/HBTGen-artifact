from tensorflow.keras import layers
from tensorflow.keras import optimizers

3
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

inputs = keras.layers.Input(shape=(784, ))    

outputs = tf.zeros([32, 10], tf.float32)

for i in range(0, 3):
    x = keras.layers.Dense(32, activation='relu', name='Model/Block' + str(i) + '/relu')(inputs) 
    x = keras.layers.Dropout(0.2, name='Model/Block' + str(i) + '/dropout')(x)
    x = keras.layers.Dense(10, activation='softmax', name='Model/Block' + str(i) + '/softmax')(x)
    outputs = keras.layers.Lambda(lambda x: x[0] + x[1], name='Model/add/add' + str(i))([outputs, x])

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary(line_length=100, positions=[.45, .58, .67, 1.])

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
model.fit(x_train, y_train,
          batch_size=32,
          epochs=5,
          validation_split=0.2,
          callbacks=[tensorboard_callback])

3
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

inputs = keras.layers.Input(shape=(784, ))

x = keras.layers.Dense(32, activation='relu', name='Model/Block0/relu')(inputs) 
x = keras.layers.Dropout(0.2, name='Model/Block0/dropout')(x)
outputs = keras.layers.Dense(10, activation='softmax', name='Model/Block0/softmax')(x)

for i in range(1, 3):
    x = keras.layers.Dense(32, activation='relu', name='Model/Block' + str(i) + '/relu')(inputs) 
    x = keras.layers.Dropout(0.2, name='Model/Block' + str(i) + '/dropout')(x)
    x = keras.layers.Dense(10, activation='softmax', name='Model/Block' + str(i) + '/softmax')(x)
    outputs = keras.layers.Lambda(lambda x: x[0] + x[1], name='Model/add/add' + str(i))([outputs, x])

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary(line_length=84, positions=[.46, .60, .69, 1.])

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
model.fit(x_train, y_train,
          batch_size=32,
          epochs=5,
          validation_split=0.2,
          callbacks=[tensorboard_callback])

outputs = tf.zeros([32, 10], tf.float32)

for i in range(0, 3):
    x = keras.layers.Dense(32, activation='relu', name='Model/Block' + str(i) + '/relu')(inputs) 
    x = keras.layers.Dropout(0.2, name='Model/Block' + str(i) + '/dropout')(x)
    x = keras.layers.Dense(10, activation='softmax', name='Model/Block' + str(i) + '/softmax')(x)
    outputs = outputs + x
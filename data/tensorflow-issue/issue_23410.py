model = Sequential()
model.add(Dense(500, input_shape = (TRAIN_SIZE, )))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(250))

model = Sequential()
model.add(Dense(500, input_shape = (TRAIN_SIZE, )))
model.add(Activation('relu'))
#model.add(Dropout(0.25))
model.add(Dense(250))

import tensorflow as tf
tf.Variable(True)

import tensorflow as tf
tf.get_variable('test_bool', 1, tf.bool)
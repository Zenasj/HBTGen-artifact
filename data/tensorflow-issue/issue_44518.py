import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

3
import tensorflow as tf


def prepare_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')
    model.build([None, 1])
    return model


def print_opt_weights(tag):
    print(tag, list(map(lambda x: x.name, model.optimizer.weights)))
    return


labels = inputs = tf.random.uniform([5, 1])
model = prepare_model()
print_opt_weights('After model creation:')
model.fit(x=inputs, y=labels, batch_size=1)
print_opt_weights('After training:')
model.save_weights('save')

print('------------Without Distribute------------')
model = prepare_model()
print_opt_weights('After model creation:')
status = model.load_weights('save')
print_opt_weights('After load weight:')
# model.fit(x=inputs, y=labels, batch_size=1)
# print_opt_weights('After retraining:')
# status.assert_consumed()

print('------------With Distribute------------')
with tf.distribute.MirroredStrategy().scope():
    model = prepare_model()
    print_opt_weights('After model creation:')
    status = model.load_weights('save')
    print_opt_weights('After load weight:')
    # model.fit(x=inputs, y=labels, batch_size=1)
    # print_opt_weights('After retraining:')
    # status.assert_consumed()


'''
The commented out lines will make optimizers in
<without distribute> and <with distribute>
have the same number of weights.

Upon `load_weights` call, all the slot variables in the optimizer
will be created when it's not using distribution strategy,
while they are not created until the model is actually retrained
with a distribution strategy in use.
'''
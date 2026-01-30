from tensorflow import keras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

print('tf_version:', tf.__version__, 'gpu available:', tf.test.is_gpu_available())
model = tf.keras.applications.ResNet50()


print('compiling model')
model.compile(optimizer='SGD', loss=tf.keras.losses.categorical_crossentropy)

print('running fit function')
x = tf.data.Dataset.from_tensors(tf.zeros([16]+model.input.shape.as_list()[1:]))
y = tf.data.Dataset.from_tensors(tf.zeros([16]+model.output.shape.as_list()[1:]))
print('x:', x, '\ny', y)
model.fit(tf.data.Dataset.zip((x,y)).repeat().shuffle(buffer_size=1),
          steps_per_epoch=10,
          verbose=0,
          callbacks=[tf.keras.callbacks.ProgbarLogger('steps')])
print('done')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

print('tf_version:', tf.__version__, 'gpu available:', tf.test.is_gpu_available())
import datetime

# copied from https://www.tensorflow.org/guide/keras/custom_callback
class MyCustomCallback(tf.keras.callbacks.Callback):

  def on_train_batch_begin(self, batch, logs=None):
    print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_train_batch_end(self, batch, logs=None):
    print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_begin(self, batch, logs=None):
    print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_end(self, batch, logs=None):
    print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))
    

model = tf.keras.applications.ResNet50()    
print('compiling model')
model.compile(optimizer='SGD', loss=tf.keras.losses.categorical_crossentropy)

print('running fit function')
x = tf.data.Dataset.from_tensors(tf.zeros([1]+model.input.shape.as_list()[1:]))
y = tf.data.Dataset.from_tensors(tf.zeros([1]+model.output.shape.as_list()[1:]))
print('x:', x, '\ny', y)
model.fit(tf.data.Dataset.zip((x,y)).repeat().shuffle(buffer_size=1),
          steps_per_epoch=2,
          verbose=0,
          callbacks=[MyCustomCallback()])
print('done')
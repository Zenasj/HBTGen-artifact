import random
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)
inputs = keras.Input(shape=(4096,), name='digits')
if tf.config.list_physical_devices('GPU'):
  print('The model will run with 4096 units on a GPU')
  num_units = 4096
else:
  # Use fewer units on CPUs so the model finishes in a reasonable amount of time
  print('The model will run with 64 units on a CPU')
  num_units = 64


dense1 = layers.Dense(num_units, activation='relu', name='dense_1')
x = dense1(inputs)
dense2 = layers.Dense(num_units, activation='relu', name='dense_2')
x = dense2(x)
dense3 = layers.Dense(num_units, activation='relu', name='dense_3')
x = dense3(x)
dense4 = layers.Dense(num_units, activation='relu', name='dense_4')
x = dense4(x)
dense5 = layers.Dense(num_units, activation='relu', name='dense_5')
x=dense5(x)
x = layers.Dense(16, name='dense_logits')(x)
outputs = layers.Activation('softmax', dtype='float32', name='predictions')(x)
print('Outputs dtype: %s' % outputs.dtype.name)
model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())

optimizer=keras.optimizers.RMSprop()
optimizer=tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
@tf.function
def train_step(x,y):
  with tf.GradientTape() as tape:
    predictions=model(x)
    loss=loss_object(y,predictions)
    scaled_loss=optimizer.get_scaled_loss(loss)
  scaled_gradients=tape.gradient(scaled_loss,model.trainable_variables)
  gradients=optimizer.get_unscaled_gradients(scaled_gradients)
  optimizer.apply_gradients(zip(gradients,model.trainable_variables))
  return loss
@tf.function
def test_step(x):
  return model(x,training=False)
x_train=tf.random.normal([1000000,num_units],dtype=tf.float32)
y_train=tf.random.uniform(shape=(1000000,),minval=0,maxval=15,dtype=tf.int64)
x_test=x_train
y_test=y_train
train_dataset=tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(512)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(512)
import time
start_time=time.perf_counter()
for epoch in range(10):
  epoch_loss_avg=tf.keras.metrics.Mean()
  test_accuracy=tf.keras.metrics.SparseCategoricalCrossentropy(name='test_accuracy')
  for x,y in train_dataset:
    loss=train_step(x,y)
    epoch_loss_avg(loss)
print("---- Training time in seconds %s --"%(time.perf_counter()-start_time))
start_time=time.perf_counter()
for x,y in test_dataset:
  test_accuracy=tf.keras.metrics.SparseCategoricalCrossentropy(name='test_accuracy')
  predictions=test_step(x)
  test_accuracy.update_state(y,predictions)
print("---- Testing time in seconds %s --"%(time.perf_counter()-start_time))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

inputs = keras.Input(shape=(4096,), name='digits')
if tf.config.list_physical_devices('GPU'):
  print('The model will run with 4096 units on a GPU')
  num_units = 4096
else:
  # Use fewer units on CPUs so the model finishes in a reasonable amount of time
  print('The model will run with 64 units on a CPU')
  num_units = 64


dense1 = layers.Dense(num_units, activation='relu', name='dense_1')
x = dense1(inputs)
dense2 = layers.Dense(num_units, activation='relu', name='dense_2')
x = dense2(x)
dense3 = layers.Dense(num_units, activation='relu', name='dense_3')
x = dense3(x)
dense4 = layers.Dense(num_units, activation='relu', name='dense_4')
x = dense4(x)
dense5 = layers.Dense(num_units, activation='relu', name='dense_5')
x=dense5(x)
x = layers.Dense(16, name='dense_logits')(x)
outputs = layers.Activation('softmax', dtype='float32', name='predictions')(x)
print('Outputs dtype: %s' % outputs.dtype.name)
model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())

optimizer=keras.optimizers.RMSprop()

loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
@tf.function
def train_step(x,y):
  with tf.GradientTape() as tape:
    predictions=model(x)
    loss=loss_object(y,predictions)
  gradients=tape.gradient(loss,model.trainable_variables)
  optimizer.apply_gradients(zip(gradients,model.trainable_variables))
  return loss
@tf.function
def test_step(x):
  return model(x,training=False)
x_train=tf.random.normal([1000000,num_units],dtype=tf.float32)
y_train=tf.random.uniform(shape=(1000000,),minval=0,maxval=15,dtype=tf.int64)
x_test=x_train
y_test=y_train
train_dataset=tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(512)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(512)
import time
start_time=time.perf_counter()
for epoch in range(10):
  epoch_loss_avg=tf.keras.metrics.Mean()
  test_accuracy=tf.keras.metrics.SparseCategoricalCrossentropy(name='test_accuracy')
  for x,y in train_dataset:
    loss=train_step(x,y)
    epoch_loss_avg(loss)
print("---- Training time in seconds %s --"%(time.perf_counter()-start_time))
start_time=time.perf_counter()
for x,y in test_dataset:
  test_accuracy=tf.keras.metrics.SparseCategoricalCrossentropy(name='test_accuracy')
  predictions=test_step(x)
  test_accuracy.update_state(y,predictions)
print("---- Testing time in seconds %s --"%(time.perf_counter()-start_time))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
import os

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


print("--- mixed precision enabled -----")
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)
inputs = keras.Input(shape=(2048,), name='digits')
if tf.config.list_physical_devices('GPU'):
  print('The model will run with 4096 units on a GPU')
  num_units =2048
else:
  # Use fewer units on CPUs so the model finishes in a reasonable amount of time
  print('The model will run with 64 units on a CPU')
  num_units = 64
dense1 = layers.Dense(num_units, activation='relu', name='dense_1')
x = dense1(inputs)
dense2 = layers.Dense(num_units, activation='relu', name='dense_2')
x = dense2(x)
dense3 = layers.Dense(num_units, activation='relu', name='dense_3')
x = dense3(x)
dense4 = layers.Dense(num_units, activation='relu', name='dense_4')
x = dense4(x)
dense5 = layers.Dense(num_units, activation='relu', name='dense_5')
x = dense5(x)

dense6 = layers.Dense(num_units, activation='relu', name='dense_6')
x = dense6(x)
# CORRECT: softmax and model output are float32
x = layers.Dense(16, name='dense_logits')(x)
outputs = layers.Activation('softmax', dtype='float32', name='predictions')(x)
print('Outputs dtype: %s' % outputs.dtype.name)
model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())

optimizer = keras.optimizers.RMSprop()
optimizer = mixed_precision.LossScaleOptimizer(optimizer)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
@tf.function
def train_step(x, y):
  with tf.GradientTape() as tape:
    predictions = model(x)
    loss = loss_object(y, predictions)
    scaled_loss = optimizer.get_scaled_loss(loss)
  scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
  gradients = optimizer.get_unscaled_gradients(scaled_gradients)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss


@tf.function
def test_step(x):
  return model(x, training=False)


x_train = tf.random.normal([1000000,2048],dtype=tf.float32)
y_train = tf.random.uniform(shape=(1000000,), minval=0, maxval=15, dtype=tf.int64)
x_test = x_train
y_test=y_train

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(512)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(512)
import time

for epoch in range(10):
  epoch_loss_avg = tf.keras.metrics.Mean()
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='test_accuracy')
  #temp=0
  for x, y in train_dataset:
    loss = train_step(x, y)
    epoch_loss_avg(loss)
    #print(temp)
    #temp=temp+1
  #print("loss and epoch",loss,epoch)
print("--- Training time in seconds %s---" % (time.perf_counter() - start_time))
start_time = time.perf_counter()
#test_scores = model.evaluate(x_test, y_test, verbose=2)
for x, y in test_dataset:
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='test_accuracy')
  predictions = test_step(x)
  test_accuracy.update_state(y, predictions)
print("--- Inference time in seconds %s---" % (time.perf_counter() - start_time))

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


inputs = keras.Input(shape=(2048,), name='digits')
if tf.config.list_physical_devices('GPU'):
  print('The model will run with 4096 units on a GPU')
  num_units = 2048
else:
  # Use fewer units on CPUs so the model finishes in a reasonable amount of time
  print('The model will run with 64 units on a CPU')
  num_units = 64
dense1 = layers.Dense(num_units, activation='relu', name='dense_1')
x = dense1(inputs)
dense2 = layers.Dense(num_units, activation='relu', name='dense_2')
x = dense2(x)

dense3 = layers.Dense(num_units, activation='relu', name='dense_3')
x = dense3(x)





dense4 = layers.Dense(num_units, activation='relu', name='dense_4')
x = dense4(x)
dense5 = layers.Dense(num_units, activation='relu', name='dense_5')
x = dense5(x)

dense6 = layers.Dense(num_units, activation='relu', name='dense_6')
x = dense6(x)




x = layers.Dense(16, name='dense_logits')(x)
outputs = layers.Activation('softmax', name='predictions')(x)
print('Outputs dtype: %s' % outputs.dtype.name)
model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())

optimizer = keras.optimizers.RMSprop()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
@tf.function
def train_step(x, y):
  with tf.GradientTape() as tape:
    predictions = model(x)
    loss = loss_object(y, predictions)
    
  gradients = tape.gradient(loss, model.trainable_variables)
 
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss


@tf.function
def test_step(x):
  return model(x, training=False)


x_train = tf.random.normal([1000000,2048],dtype=tf.float32)
y_train = tf.random.uniform(shape=(1000000,), minval=0, maxval=15, dtype=tf.int64)
x_test = x_train
y_test=y_train

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(512)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(512)
import time
start_time = time.perf_counter()

for epoch in range(10):
  epoch_loss_avg = tf.keras.metrics.Mean()
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='test_accuracy')
  for x, y in train_dataset:
    loss = train_step(x, y)
    epoch_loss_avg(loss)
  #print("loss and epoch",loss,epoch)
print("--- Training time in seconds %s---" % (time.perf_counter() - start_time))
start_time = time.perf_counter()
#test_scores = model.evaluate(x_test, y_test, verbose=2)
for x, y in test_dataset:
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='test_accuracy')
  predictions = test_step(x)
  test_accuracy.update_state(y, predictions)
print("--- Inference time in seconds %s---" %(time.perf_counter() - start_time))
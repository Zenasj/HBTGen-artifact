import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

hits += tf.reduce_sum(tf.where((output - target_batch + x_slice) < 0.01, 1, 0)).numpy() # this line causes the error
  # hits += tf.reduce_sum(tf.where(output - target_batch < 0.01, 1, 0)).numpy() # if x_slice is omitted script works fine

import tensorflow as tf

batch_size = 128

input = tf.keras.Input(shape=(None, 1))
x = tf.keras.layers.Dense(1)(input)
output = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs=input, outputs=output)

# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 2000
inputs = tf.random.normal([NUM_EXAMPLES])
noise = tf.random.normal([NUM_EXAMPLES])
outputs = inputs * 3 + 2 + noise

training_inputs = tf.reshape(inputs[:1500], (1500, 1))
training_outputs = tf.reshape(outputs[:1500], (1500, 1))
training_inputs = tf.data.Dataset.from_tensor_slices(training_inputs).batch(batch_size)
training_outputs = tf.data.Dataset.from_tensor_slices(training_outputs).batch(batch_size)
test_inputs = tf.reshape(inputs[1500:], (500, 1))
test_outputs = tf.reshape(outputs[1500:], (500, 1))
test_inputs = tf.data.Dataset.from_tensor_slices(test_inputs).batch(batch_size)
test_outputs = tf.data.Dataset.from_tensor_slices(test_outputs).batch(batch_size)


def loss(model, inputs, targets):
  outputs = model(inputs)
  output = outputs[:, 0]  # take the first output (in general model can have several outputs)
  global x
  x_slice = x[:, 0]
  error = output - targets + x_slice
  return tf.reduce_mean(tf.square(error))


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
epoch = 3
for i in range(epoch):
  for input_batch, target_batch in zip(training_inputs, training_outputs):
    with tf.GradientTape() as tape:
      loss_value = loss(model, input_batch, target_batch)
      grads = tape.gradient(loss_value, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
  print('epoch #:', i)


hits = 0
total = 0
for input_batch, target_batch in zip(test_inputs, test_outputs):
  outputs = model(input_batch)
  output = outputs[:, 0]  # take the first output (in general model can have several outputs)
  x_slice = x[:, 0]
  hits += tf.reduce_sum(tf.where((output - target_batch + x_slice) < 0.01, 1, 0)).numpy() # this line causes the error
  # hits += tf.reduce_sum(tf.where(output - target_batch < 0.01, 1, 0)).numpy() # if x_slice is omitted script works fine
  total += input_batch.shape[0]

print(hits)
print('Accuracy: ', hits/total)
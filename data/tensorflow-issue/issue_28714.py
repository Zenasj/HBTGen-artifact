from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import numpy as np
from time import perf_counter

def get_uncompiled_model():
  inputs = keras.Input(shape=(784,), name='digits')
  x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
  x = layers.Dense(64, activation='relu', name='dense_2')(x)
  outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
  model = keras.Model(inputs=inputs, outputs=outputs)
  return model

def get_compiled_model():
  model = get_uncompiled_model()
  model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
  return model

# Load a toy dataset for the sake of this example
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data (these are Numpy arrays)
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(64)

# model = get_compiled_model()

# # keras train
# model.fit(train_dataset, epochs=3)

# manual train
model = get_uncompiled_model()
optimizer = keras.optimizers.RMSprop(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy()

@tf.function
def train():
    # Iterate over epochs.
    for epoch in range(3):
      print('Start of epoch %d' % (epoch))

      for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
          logits = model(x_batch_train)  # Logits for this minibatch
          loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
s = perf_counter()
train()
print(perf_counter() - s)

model = get_compiled_model()
s = perf_counter()
model.fit(train_dataset, epochs=3)
print(perf_counter() - s)

@tf.function
def train_batch(x_batch_train, y_batch_train):
    with tf.GradientTape() as tape:
        logits = model(x_batch_train)  # Logits for this minibatch
        loss_value = loss_fn(y_batch_train, logits)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# Iterate over epochs.
for epoch in range(3):
    print('Start of epoch %d' % (epoch))
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        train_batch(x_batch_train, y_batch_train)
        
s = perf_counter()
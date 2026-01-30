from tensorflow.keras import layers
from tensorflow.keras import optimizers

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load MNIST dataset
# Mine network to download this is too slow, so I download it handly. You can also use this command line:

# mnist = keras.datasets.mnist
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

def load_mnist(path):
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)
(train_images, train_labels), (test_images, test_labels) = load_mnist(path='mnist.npz')

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model architecture.
model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(28, 28)),
  keras.layers.Reshape(target_shape=(28, 28, 1)),
  keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(10)
])

# Train the digit classification model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
  train_images,
  train_labels,
  epochs=1,
  validation_split=0.1,
)

import tensorflow_model_optimization as tfmot

quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

q_aware_model.summary()
train_images_subset = train_images[0:1000] # out of 60000
train_labels_subset = train_labels[0:1000]
q_aware_model.fit(train_images_subset, train_labels_subset,
                  batch_size=500, epochs=1, validation_split=0.1)
_, baseline_model_accuracy = model.evaluate(
    test_images, test_labels, verbose=0)

_, q_aware_model_accuracy = q_aware_model.evaluate(
   test_images, test_labels, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)
print('Quant test accuracy:', q_aware_model_accuracy)

# q_aware stands for for quantization aware.
q_aware_model2 = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model2.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

q_aware_model2.summary()

batch_size = 500
train_dataset = tf.data.Dataset.from_tensor_slices((train_images_subset, train_labels_subset))
train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=False)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

for epoch in range(1):
    for x, y in train_dataset:
        with tf.GradientTape() as tape:
            preds = q_aware_model2(x)
            loss = loss_fn(y, preds)
        grads = tape.gradient(loss, q_aware_model2.trainable_variables)
        optimizer.apply_gradients(zip(grads, q_aware_model2.trainable_variables))
        
_, baseline_model_accuracy = model.evaluate(
    test_images, test_labels, verbose=0)

_, q_aware_model_accuracy = q_aware_model2.evaluate(
   test_images, test_labels, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)
print('Quant test accuracy:', q_aware_model_accuracy)

# q_aware stands for for quantization aware.
q_aware_model2 = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model2.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

q_aware_model2.summary()

batch_size = 500
train_dataset = tf.data.Dataset.from_tensor_slices((train_images_subset, train_labels_subset))
train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=False)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

# for epoch in range(1):
#     for x, y in train_dataset:
#         with tf.GradientTape() as tape:
#             preds = q_aware_model2(x)
#             loss = loss_fn(y, preds)
#         grads = tape.gradient(loss, q_aware_model2.trainable_variables)
#         optimizer.apply_gradients(zip(grads, q_aware_model2.trainable_variables))
        
_, baseline_model_accuracy = model.evaluate(
    test_images, test_labels, verbose=0)

_, q_aware_model_accuracy = q_aware_model2.evaluate(
   test_images, test_labels, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)
print('Quant test accuracy:', q_aware_model_accuracy)
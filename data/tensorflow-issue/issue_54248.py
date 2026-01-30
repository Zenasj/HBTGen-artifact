from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow_datasets as tfds
import tempfile
import os
import shutil

# Load MNIST Datasets
print("Load MNIST Dataset")
mnist_datasets, ds_info = tfds.load('mnist', as_supervised=True, with_info=True)
train_ds = mnist_datasets['train']
test_ds = mnist_datasets['test']

batch_size = 32

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

try:
    autotune_opt = tf.data.AUTOTUNE
except:
    autotune_opt = tf.data.experimental.AUTOTUNE

train_ds = train_ds.map(
    normalize_img, num_parallel_calls=autotune_opt)
train_ds = train_ds.cache()
train_ds = train_ds.shuffle(ds_info.splits['train'].num_examples)
train_ds = train_ds.batch(batch_size)

test_ds = test_ds.map(
    normalize_img, num_parallel_calls=autotune_opt)
test_ds = test_ds.batch(batch_size)
test_ds = test_ds.cache()
test_ds = test_ds.prefetch(autotune_opt)

# Define function to create functional keras model
def create_keras_model(layer_dims):
    inp = tf.keras.layers.Input((28,28,1), name="Input_Name")
    last_layer = inp
    last_layer = tf.keras.layers.Reshape((28*28*1,), name="Input_Reshape")(last_layer)
    for i in range(len(layer_dims)):
        dim = layer_dims[i]
        last_layer = tf.keras.layers.Dense(dim, name=f"Layer_{i}")(last_layer)
    outp = tf.keras.layers.Dense(10, name="Output_Layer", activation='sigmoid')(last_layer)

    return tf.keras.Model(inputs=inp, outputs=outp)

# Create and train initial model:

print("Create and train initial keras model")
mdl_1 = create_keras_model([32,32,32])

mdl_1.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=tf.keras.metrics.SparseCategoricalAccuracy())

mdl_1.fit(train_ds, epochs=1)

# Create function to test models
def eval_on_test(mdl):
    num_matches = 0
    total_examples = 0

    for X, Y in test_ds:
        # Eval model
        Y_eval = tf.argmax(mdl(X),axis=1).numpy()

        # Count
        total_examples += len(Y_eval)
        num_matches += (Y.numpy() == Y_eval).sum()

    return num_matches/total_examples

print("Initial model performance")
print(eval_on_test(mdl_1))

# Save weights to disk
mdl_weight_file = 'weights.hdf5'
mdl_1.save_weights(mdl_weight_file)

# Create new model and load from a temporary file
mdl_2 = create_keras_model([32,32,32])

with tempfile.NamedTemporaryFile('w+b') as temp_f:
    # Copy content to temporary file
    print("Copy weights to temp file")
    with open(mdl_weight_file, 'rb') as f:
        temp_f.write(f.read())

    # Flush to disk
    temp_f.flush()

    print("Load weights from temp file")
    # Load weights from temp file
    mdl_2.load_weights(temp_f.name)

print("temp file gone")

print("Loaded model performance")
print(eval_on_test(mdl_2))
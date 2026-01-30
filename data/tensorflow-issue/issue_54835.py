from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops.numpy_ops import np_config

# Create a dictionary describing the features.
observ = {
        'data' : tf.io.FixedLenFeature([104], tf.int64),
        'label' : tf.io.FixedLenFeature([], tf.int64) }

# Parse the input tf.train.Example proto using the dictionary above.
def _map_function_1(example_proto):
  return tf.io.parse_single_example(example_proto, observ)

def split_data_label(sample):
    return sample['data'].astype('int16'), sample['label'].astype('int16')

# Import End game data
def load_data() :
    dataset = tf.data.TFRecordDataset('EndGsh.tfrecords')
    parsed_dataset = dataset.map(_map_function_1)
    parsed_and_split_datased = parsed_dataset.map(split_data_label)

    DATASET_SIZE = 10000 # 2836995
    train_size = int(0.7 * DATASET_SIZE)
    val_size = int(0.10 * DATASET_SIZE)
    test_size = int(0.20 * DATASET_SIZE)

    parsed_and_split_datased = parsed_and_split_datased.shuffle(DATASET_SIZE)

    train_dataset = parsed_and_split_datased.take(train_size)
    tAv_dataset = parsed_and_split_datased.skip(train_size)
    val_dataset = tAv_dataset.take(val_size)
    test_dataset = tAv_dataset.skip(val_size)

    return train_dataset, test_dataset, val_dataset

# Define model
np_config.enable_numpy_behavior()

model = keras.Sequential()
model.add(keras.Input(shape=(104,)))
model.add(layers.Dense(64, activation="relu", name="end_game_in"))
model.add(layers.Dense(32, activation="relu", name="end_game_h1"))
model.add(layers.Dense(16, activation="relu", name="end_game_h2"))
model.add(layers.Dense(1, activation="relu", name="end_game_out"))
model.summary()

train_dataset , test_dataset, val_dataset = load_data()

model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

print("Fit model on training data")
x_train =[]
y_train = []
count = 0
for it in train_dataset :
    x , y = it
    x_train.append(x.astype("float32"))
    y_train.append(y.astype("float32"))
    count += 1
    if count % 5000 == 0:
        print("Train: ", count)

x_val =[]
y_val = []
count = 0
for it in val_dataset :
    x , y = it
    x_val.append(x.astype("float32"))
    y_val.append(y.astype("float32"))
    count += 1
    if count % 5000 == 0:
        print("Val: ", count)

history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=5,
    verbose=2,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
)

# rest of script happens after the error
history.history

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")

x_test =[]
y_test = []
count = 0
for it in val_dataset :
    x , y = it
    x_test.append(x.astype("float32"))
    y_test.append(y.astype("float32"))
    count += 1
    if count % 5000 == 0:
        print("Test: ", count)

results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using predict
print("Generate predictions for 3 samples")
predictions = model.predict(x_test[:3])
print("predictions shape:", predictions.shape)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf

input = tf.keras.layers.Input(shape=(2), dtype=tf.float32)
output = tf.keras.layers.Dense(1, activation="relu")(input)
model = tf.keras.Model(input, output)

learning_rate_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay([10], [0.5, 0.25])
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_schedule)
model.compile(optimizer=optimizer, loss="mse")

class ValidationCallback(tf.keras.callbacks.Callback):

  def __init__(self, expected):
    self.expected = expected
    self.actual = []

  def on_train_batch_end(self, batch, logs=None):
    self.actual.append(tf.keras.backend.get_value(self.model.optimizer.learning_rate(self.model.optimizer._iterations)).item())

  def on_train_end(self, logs=None):
    if self.actual == self.expected:
      print("Good")
    else:
      print(f"Bug: Expected {self.expected}, actual {self.actual}")

values = tf.range(20, dtype=tf.float32)
inputs = tf.stack([values, values], axis=1)
outputs = values * 2.0
dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
dataset = dataset.batch(1)

# Call fit() for steps 1-20.
print("Test 1")
model.fit(dataset, epochs=2, initial_epoch=0, steps_per_epoch=10, callbacks=[ValidationCallback([0.5] * 10 + [0.25] * 10)], verbose=0)

# Call fit() for steps 1-20 again (bug: the optimizer just retains its iterations, rather than calculating from epochs + initial_epoch + steps_per_epoch).
print("Test 2")
model.fit(dataset, epochs=2, initial_epoch=0, steps_per_epoch=10, callbacks=[ValidationCallback([0.5] * 10 + [0.25] * 10)], verbose=0)

# Call fit() for steps 1-20 a third time (work around the bug by resetting iterations).
print("Test 3")
model.optimizer.iterations.assign(0, read_value=False)
model.fit(dataset, epochs=2, initial_epoch=0, steps_per_epoch=10, callbacks=[ValidationCallback([0.5] * 10 + [0.25] * 10)], verbose=0)

# Call fit() for steps 11-20 (bug: should calculate iterations from epochs + initial_epoch + steps_per_epoch - very important when resuming training from a checkpoint).
print("Test 4")
model.optimizer.iterations.assign(0, read_value=False)
model.fit(dataset, epochs=2, initial_epoch=1, steps_per_epoch=10, callbacks=[ValidationCallback([0.25] * 10)], verbose=0)

# Call fit() for steps 11-20 (work around the bug by manually initializing iterations)
print("Test 5")
initial_epoch = 1
steps_per_epoch = 10
model.optimizer.iterations.assign(initial_epoch * steps_per_epoch, read_value=False)
model.fit(dataset, epochs=2, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, callbacks=[ValidationCallback([0.25] * 10)], verbose=0)
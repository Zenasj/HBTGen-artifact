import random
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

#optimizer_type = SGD  # this works
optimizer_type = Adam  # this does not work

np.random.seed(42)

x = np.array(range(100), dtype=np.float32).reshape(-1, 1)
x_batched = np.split(x, 2)
y = 2 * x + 1 + np.random.normal(size=x.shape[0]).reshape(-1, 1)
y_batched = np.split(y, 2)

loss_fn = MeanSquaredError()


class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.d1 = Dense(5, activation='relu', dtype=tf.float32)
        self.d2 = Dense(1, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        return self.d2(self.d1(inputs))


def train_step(model, loss_fn, optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value


def optimize(model, loss_fn, optimizer, epochs):
    losses = []
    for epoch in range(epochs):
        for x_batch, y_batch in zip(x_batched, y_batched):
            loss_value = train_step(model, loss_fn, optimizer, x_batch,
                                    y_batch)
            losses.append(loss_value.numpy())
    return losses


# train 4 epochs

tf.random.set_seed(42)
model = MyModel()
optimizer = optimizer_type(learning_rate=1e-3)
losses = optimize(model, loss_fn, optimizer, epochs=4)
pred = model(x_batched[0])

# train 2 epochs

tf.random.set_seed(42)
model = MyModel()
optimizer = optimizer_type(learning_rate=1e-3)
losses2 = optimize(model, loss_fn, optimizer, epochs=2)
model.save_weights('weights')
with open('opt.pkl', 'wb') as f:
    pickle.dump(optimizer, f)

# load and train 2 more epochs

model = MyModel()
model.load_weights('weights')
with open('opt.pkl', 'rb') as f:
    optimizer = pickle.load(f)
losses2.extend(optimize(model, loss_fn, optimizer, epochs=2))
pred2 = model(x_batched[0])

print(losses)
print(losses2)
print(pred)
print(pred2)

assert losses == losses2
assert np.all(pred == pred2)

import pickle

import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import Dense, Layer
from keras.losses import MeanSquaredError
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop, Adadelta, Adagrad, Adamax, Nadam

# optimizer_type = SGD      # this works
optimizer_type = Adam       # this does not work not using checkpoint, works using checkpoint
# optimizer_type = RMSprop      # this does not work not using checkpoint, works using checkpoint
# optimizer_type = Adadelta     # this does not work not using checkpoint, works using checkpoint
# optimizer_type = Adagrad      # this does not work not using checkpoint, works using checkpoint
# optimizer_type = Nadam        # this does not work not using checkpoint, works using checkpoint

np.random.seed(42)

x = np.array(range(100), dtype=np.float32).reshape(-1, 1)
x_batched = np.split(x, 2)
y = 2 * x + 1 + np.random.normal(size=x.shape[0]).reshape(-1, 1)
y_batched = np.split(y, 2)

loss_fn = MeanSquaredError()

class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.d1 = Dense(5, activation='relu', dtype=tf.float32)
        self.d2 = Dense(1, dtype=tf.float32)

    def call(self, inputs):
        return self.d2(self.d1(inputs))


def train_step(model, loss_fn, optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value


def optimize(model, loss_fn, optimizer, epochs):
    losses = []
    for epoch in range(epochs):
        for x_batch, y_batch in zip(x_batched, y_batched):
            loss_value = train_step(model, loss_fn, optimizer, x_batch,
                                    y_batch)
            losses.append(loss_value.numpy())
    return losses


# train 4 epochs

folder_checkpoint = './study/test_save_optimizer/checkpoints' # it should define in each environment

tf.random.set_seed(42)
model = MyModel()
optimizer = optimizer_type(learning_rate=1e-3)
losses = optimize(model, loss_fn, optimizer, epochs=4)
pred = model(x_batched[0])

# train 2 epochs

tf.random.set_seed(42)
model = MyModel()
optimizer = optimizer_type(learning_rate=1e-3)
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
manager = tf.train.CheckpointManager(checkpoint, folder_checkpoint, max_to_keep=1)

losses2 = optimize(model, loss_fn, optimizer, epochs=2)

save_path = manager.save()
print("Saved checkpoint {}".format(save_path))

# Previous Trials
# model.save_weights('./weights')
# with open('./optimizer_weights.pkl', 'wb') as f:
#     pickle.dump(optimizer.get_weights(), f)


# load and train 2 more epochs
model = MyModel()
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
manager = tf.train.CheckpointManager(checkpoint, folder_checkpoint, max_to_keep=1)

# # Previous Trials
# model.load_weights('./weights')
# with open('./optimizer_weights.pkl', 'rb') as f:
#     optimizer.set_weights(pickle.load(f))

checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

losses2.extend(optimize(model, loss_fn, optimizer, epochs=2))
pred2 = model(x_batched[0])

print(losses)
print(losses2)
print(pred)
print(pred2)

assert losses == losses2
assert np.all(pred == pred2)

"""ISSUE: https://github.com/tensorflow/tensorflow/issues/41053
"""
import pickle
import json
import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import Dense, Layer
from keras.losses import MeanSquaredError
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop, Adadelta, Adagrad, Adamax, Nadam

# optimizer_type = SGD      # this works
# optimizer_type = Adam       # this does not work not using checkpoint, works using checkpoint
# optimizer_type = RMSprop      # this does not work not using checkpoint, works using checkpoint
# optimizer_type = Adadelta     # this does not work not using checkpoint, works using checkpoint
# optimizer_type = Adagrad      # this does not work not using checkpoint, works using checkpoint
optimizer_type = Nadam        # this does not work not using checkpoint, works using checkpoint

np.random.seed(42)

x = np.array(range(100), dtype=np.float32).reshape(-1, 1)
x_batched = np.split(x, 2)
y = 2 * x + 1 + np.random.normal(size=x.shape[0]).reshape(-1, 1)
y_batched = np.split(y, 2)

loss_fn = MeanSquaredError()

class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.d1 = Dense(5, activation='relu', dtype=tf.float32)
        self.d2 = Dense(1, dtype=tf.float32)

    def call(self, inputs):
        return self.d2(self.d1(inputs))


def train_step(model, loss_fn, optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value


def optimize(model, loss_fn, optimizer, epochs):
    losses = []
    for epoch in range(epochs):
        for x_batch, y_batch in zip(x_batched, y_batched):
            loss_value = train_step(model, loss_fn, optimizer, x_batch,
                                    y_batch)
            losses.append(loss_value.numpy())
    return losses

# train 4 epochs

tf.random.set_seed(42)
model = MyModel()
optimizer = optimizer_type(learning_rate=1e-3)
losses = optimize(model, loss_fn, optimizer, epochs=4)
pred = model(x_batched[0])

# train 2 epochs

tf.random.set_seed(42)
model = MyModel()
optimizer = optimizer_type(learning_rate=1e-3)


losses2 = optimize(model, loss_fn, optimizer, epochs=2)

model.save_weights('./weights')
with open('./optimizer_weights.pkl', 'wb') as f:
    pickle.dump(optimizer.get_weights(), f)


# load and train 2 more epochs
model = MyModel()
optimizer = optimizer_type(learning_rate=1e-3)  # [NEW] Redefine optimizer

model.compile(optimizer=optimizer, loss='mse')
# model.build(input_shape=(None, 1))
model.fit(x_batched[0], y_batched[0])   # [NEW] Compile first time for setting the weight shape

model.load_weights('./weights')
with open('./optimizer_weights.pkl', 'rb') as f:
    optimizer.set_weights(pickle.load(f))



losses2.extend(optimize(model, loss_fn, optimizer, epochs=2))
pred2 = model(x_batched[0])

print(losses)
print(losses2)
print(pred)
print(pred2)

assert losses == losses2
assert np.all(pred == pred2)
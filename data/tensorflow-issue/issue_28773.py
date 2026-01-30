import random
from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.python.keras import testing_utils
import numpy as np
from tensorflow.python import keras
from tensorflow.python.framework import ops

class TestCallback(keras.callbacks.Callback):
    def set_model(self, model):
        # Check the model operations for the optimizer operations that
        # the _make_train_function adds under a named scope for the
        # optimizer. This ensures the full model is populated before the
        # set_model callback is called.
        optimizer_name_scope = 'training/' + model.optimizer.__class__.__name__
        graph_def = ops.get_default_graph().as_graph_def()
        for node in graph_def.node:
          if node.name.startswith(optimizer_name_scope):
            return
        raise RuntimeError('The optimizer operations are not present in the '
                           'model graph when the Callback.set_model function '
                           'is called')
np.random.seed(1337)

def generator():
    x = np.random.randn(10, 100).astype(np.float32)
    y = np.random.randn(10, 10).astype(np.float32)
    while True:
        yield x, y

model = testing_utils.get_small_sequential_mlp(
  num_hidden=10, num_classes=10, input_dim=100)
model.compile(
  loss='categorical_crossentropy',
  optimizer='sgd',
  metrics=['accuracy'])
model.fit_generator(
  generator(),
  steps_per_epoch=2,
  epochs=1,
  validation_data=generator(),
  validation_steps=2,
  callbacks=[TestCallback()],
  verbose=0)

from tensorflow.python.keras import testing_utils
import numpy as np
from tensorflow.python import keras
from tensorflow.python.framework import ops

class TestCallback(keras.callbacks.Callback):
    def set_model(self, model):
        # Check the model operations for the optimizer operations that
        # the _make_train_function adds under a named scope for the
        # optimizer. This ensures the full model is populated before the
        # set_model callback is called.
        optimizer_name_scope = 'training/' + model.optimizer.__class__.__name__
        graph_def = ops.get_default_graph().as_graph_def()
        for node in graph_def.node:
          if node.name.startswith(optimizer_name_scope):
            return
        raise RuntimeError('The optimizer operations are not present in the '
                           'model graph when the Callback.set_model function '
                           'is called')
np.random.seed(1337)


model = testing_utils.get_small_sequential_mlp(
  num_hidden=10, num_classes=10, input_dim=100)
model.compile(
  loss='categorical_crossentropy',
  optimizer='sgd',
  metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
    train_samples=3,
    test_samples=1,
    input_shape=(100,),
    num_classes=10)
y_test = keras.utils.to_categorical(y_test)
y_train = keras.utils.to_categorical(y_train)

model.fit(x_train, y_train, batch_size=1, epochs=1,
          callbacks=[TestCallback()])

class TestCallback(keras.callbacks.Callback):
    def set_model(self, model):
        # Check the model operations for the optimizer operations that
        # the _make_train_function adds under a named scope for the
        # optimizer. This ensures the full model is populated before the
        # set_model callback is called.
        optimizer_op_str = "gradients/loss"
        graph_def = ops.get_default_graph().as_graph_def()
        for node in graph_def.node:
          if optimizer_op_str in node.name:
            return
        raise RuntimeError('The optimizer operations are not present in the '
                           'model graph when the Callback.set_model function '
                           'is called')
np.random.seed(1337)

def generator():
    x = np.random.randn(10, 100).astype(np.float32)
    y = np.random.randn(10, 10).astype(np.float32)
    while True:
        yield x, y

# ====================================
# << See comment on unrelated model >>
# ====================================

model = testing_utils.get_small_sequential_mlp(
  num_hidden=2, num_classes=10, input_dim=100)
model.compile(
  loss='categorical_crossentropy',
  optimizer='sgd',
  metrics=['accuracy'])
model.fit_generator(
  generator(),
  steps_per_epoch=2,
  epochs=1,
  validation_data=generator(),
  validation_steps=2,
  callbacks=[TestCallback()],
  verbose=0)

unrelated_input = keras.Input(shape=(100,))
unrelated_model = keras.models.Model(unrelated_input, keras.layers.Dense(10)(unrelated_input))
unrelated_model.compile(loss="mse", optimizer="sgd")
x = np.random.randn(10, 100).astype(np.float32)
y = np.random.randn(10, 10).astype(np.float32)
unrelated_model.fit(x, y, steps_per_epoch=2, epochs=1, verbose=0, callbacks=[TestCallback()])
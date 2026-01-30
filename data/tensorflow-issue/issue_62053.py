import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf
print(tf.__version__)
import wandb
from wandb.keras import WandbModelCheckpoint
from wandb.keras import WandbCallback

run = wandb.init(project="keras")

x = np.random.randint(255, size=(100, 28, 28, 1))
y = np.random.randint(10, size=(100,))

dataset = (x, y)


def get_model():
    m = tf.keras.Sequential()
    m.add(tf.keras.layers.Conv2D(3, 3, activation="relu", input_shape=(28, 28, 1)))
    m.add(tf.keras.layers.Flatten())
    m.add(tf.keras.layers.Dense(10, activation="softmax"))
    return m


model = get_model()
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="sgd",
    metrics=["accuracy"],
)

model.fit(
    x,
    y,
    epochs=5,
    validation_data=(x, y),
    callbacks=[
        WandbCallback(
            save_model=False,
            log_gradients=True,
            training_data=(x,y)
        )
    ],
)

import tensorflow as tf
print(tf.__version__)
import wandb
import numpy as np

_training_data_x = np.random.randint(255, size=(100, 28, 28, 1))
_training_data_y = np.random.randint(10, size=(100,))


def get_model():
    m = tf.keras.Sequential()
    m.add(tf.keras.layers.Conv2D(3, 3, activation="relu", input_shape=(28, 28, 1)))
    m.add(tf.keras.layers.Flatten())
    m.add(tf.keras.layers.Dense(10, activation="softmax"))
    return m

model = get_model()
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="sgd",
    metrics=["accuracy"],
)


def _get_custom_optimizer_parent_class():
    from pkg_resources import parse_version

    if parse_version(tf.__version__) >= parse_version("2.9.0"):
        custom_optimizer_parent_class = tf.keras.optimizers.legacy.Optimizer
    else:
        custom_optimizer_parent_class = tf.keras.optimizers.Optimizer

    return custom_optimizer_parent_class


_custom_optimizer_parent_class = _get_custom_optimizer_parent_class()
print(_custom_optimizer_parent_class)


class _CustomOptimizer(_custom_optimizer_parent_class):
    def __init__(self):
        super().__init__(name="CustomOptimizer")
        self._resource_apply_dense = tf.function(self._resource_apply_dense)
        self._resource_apply_sparse = tf.function(self._resource_apply_sparse)
        tf.print(self._resource_apply_dense)

    def _resource_apply_dense(self, grad, var):
        var.assign(grad)

    # this needs to be implemented to prevent a NotImplementedError when
    # using Lookup layers.
    def _resource_apply_sparse(self, grad, var, indices):
        pass

    def get_config(self):
        return super().get_config()


class _GradAccumulatorCallback(tf.keras.callbacks.Callback):
    """Accumulates gradients during a fit() call when used in conjunction with the CustomOptimizer above."""

    def set_model(self, model):
        super().set_model(model)
        self.og_weights = model.get_weights()
        self.grads = [np.zeros(tuple(w.shape)) for w in model.trainable_weights]

    def on_batch_end(self, batch, logs=None):
        for g, w in zip(self.grads, self.model.trainable_weights):
            g += w.numpy()
        self.model.set_weights(self.og_weights)

    def get_grads(self):
        return [g.copy() for g in self.grads]


inputs = model.inputs
print(inputs)
outputs = model(inputs)
grad_acc_model = tf.keras.models.Model(inputs, outputs)
grad_acc_model.compile(loss=model.loss, optimizer=_CustomOptimizer())

_grad_accumulator_model = grad_acc_model
_grad_accumulator_model.summary()

_grad_accumulator_callback = _GradAccumulatorCallback()


_grad_accumulator_model.fit(
    _training_data_x,
    _training_data_y,
    verbose=0,
    callbacks=[_grad_accumulator_callback],
)

weights = model.trainable_weights
grads = _grad_accumulator_callback.grads
print(weights)

metrics = {}
for weight, grad in zip(weights, grads):
    metrics[
        "gradients/" + weight.name.split(":")[0] + ".gradient"
    ] = wandb.Histogram(grad)

print(metrics)
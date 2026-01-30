import math
import random
from tensorflow.keras import optimizers

model.build()
optimizer = get_patched_optimizer(optimizer, n, trainable_variables)
model.compile(optimizer=optimizer)

class _GradientAccumulationPatch:

    def __init__(
        self,
        n: int,
        orig_apply_gradients,
        trainable_variables
    ):
        self.n = tf.constant(n, dtype=tf.int64)
        policy = tf.keras.mixed_precision.global_policy()
        self.variable_dtype = policy.variable_dtype
        self.accu_gradients = [
            tf.Variable(
                tf.zeros(g.shape, dtype=g.dtype),
            ) for g in trainable_variables
        ]

        self._current_step = tf.Variable(0, dtype=tf.int64)
        self._orig_apply_gradients = orig_apply_gradients

    def apply_gradients(self, grads_and_vars, *args, **kwargs):

        can_apply = self._can_apply_on_next_step()
        # 1.0 whenever we want to apply gradients; 0.0 otherwise
        apply = tf.cast(can_apply, dtype=self.variable_dtype)
        # Will be 0.0 if apply is 1.0 and vice versa
        keep = tf.cast(tf.logical_not(can_apply), dtype=self.variable_dtype)

        grads_and_vars = list(grads_and_vars)
        gradients = [grad for (grad, _) in grads_and_vars]
        trainable_variables = [var for (_, var) in grads_and_vars]

        # Accumulate gradients
        for i, grad in enumerate(gradients):
            # FIXME should be assign_add()
            self.accu_gradients[i].assign(grad / tf.cast(self.n, dtype=grad.dtype))

        # Multiply each gradient with our apply-signal
        final_gradients = [grad * apply for grad in self.accu_gradients]

        self._orig_apply_gradients(zip(final_gradients, trainable_variables), *args, **kwargs)

        # This will reset our buffer whenever "keep" is 0.0
        for g in self.accu_gradients:
            g.assign(g * keep)

    def apply_accu_gradients(self, trainable_variables, *args, **kwargs):

        # Call the original apply_gradients() function
        self._orig_apply_gradients(zip(self.accu_gradients, trainable_variables), *args, **kwargs)

        # Reset all accumulated gradients to zero
        for i in range(len(self.accu_gradients)):
            self.accu_gradients[i].assign(tf.zeros_like(trainable_variables[i]))

    def _can_apply_on_next_step(self):
        """
        :return: True if gradients should be applied; False otherwise.
        """
        # Increment (always do this first)
        self._current_step.assign_add(1)
        count_mod_steps = tf.math.mod(self._current_step, self.n)
        return tf.equal(count_mod_steps, 0)


def get_patched_optimizer(optimizer, n, trainable_variables):
    """Patch optimizer for gradient accumulation.

    :param optimizer:
        The optimizer to patch.
    :param n:
        The number of accumulation steps before applying gradients.
    :param trainable_variables:
        Trainable parameters of the model
    :return:
        A patched patched optimizer for gradient accumulation.
    """
    accumulator = _GradientAccumulationPatch(
        n=n,
        orig_apply_gradients=optimizer.apply_gradients,
        trainable_variables=trainable_variables
    )

    # Replace the original function
    optimizer.apply_gradients = accumulator.apply_gradients

    return optimizer

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class _GradientAccumulationPatch:

    def __init__(
        self,
        n: int,
        orig_apply_gradients,
        trainable_variables
    ):
        self.n = tf.constant(n, dtype=tf.int64)
        policy = tf.keras.mixed_precision.global_policy()
        self.variable_dtype = policy.variable_dtype
        self.accu_gradients = [
            tf.Variable(
                tf.zeros(g.shape, dtype=g.dtype),
            ) for g in trainable_variables
        ]

        self._current_step = tf.Variable(0, dtype=tf.int64)
        self._orig_apply_gradients = orig_apply_gradients

    def apply_gradients(self, grads_and_vars, *args, **kwargs):

        can_apply = self._can_apply_on_next_step()
        # 1.0 whenever we want to apply gradients; 0.0 otherwise
        apply = tf.cast(can_apply, dtype=self.variable_dtype)
        # Will be 0.0 if apply is 1.0 and vice versa
        keep = tf.cast(tf.logical_not(can_apply), dtype=self.variable_dtype)

        grads_and_vars = list(grads_and_vars)
        gradients = [grad for (grad, _) in grads_and_vars]
        trainable_variables = [var for (_, var) in grads_and_vars]

        # Accumulate gradients
        for i, grad in enumerate(gradients):
            self.accu_gradients[i].assign_add(grad / tf.cast(self.n, dtype=self.variable_dtype))

        # Multiply each gradient with our apply-signal
        final_gradients = [grad * apply for grad in self.accu_gradients]

        self._orig_apply_gradients(zip(final_gradients, trainable_variables), *args, **kwargs)

        # This will reset our buffer whenever "keep" is 0.0
        for g in self.accu_gradients:
            g.assign(g * keep)

    def apply_accu_gradients(self, trainable_variables, *args, **kwargs):

        # Call the original apply_gradients() function
        self._orig_apply_gradients(zip(self.accu_gradients, trainable_variables), *args, **kwargs)

        # Reset all accumulated gradients to zero
        for i in range(len(self.accu_gradients)):
            self.accu_gradients[i].assign(tf.zeros_like(trainable_variables[i]))

    def _can_apply_on_next_step(self):
        """
        :return: True if gradients should be applied; False otherwise.
        """
        # Increment (always do this first)
        self._current_step.assign_add(1)
        count_mod_steps = tf.math.mod(self._current_step, self.n)
        return tf.equal(count_mod_steps, 0)


def get_patched_optimizer(optimizer, n, trainable_variables):
    """Patch optimizer for gradient accumulation.

    :param optimizer:
        The optimizer to patch.
    :param n:
        The number of accumulation steps before applying gradients.
    :param trainable_variables:
        Trainable parameters of the model
    :return:
        A patched patched optimizer for gradient accumulation.
    """
    accumulator = _GradientAccumulationPatch(
        n=n,
        orig_apply_gradients=optimizer.apply_gradients,
        trainable_variables=trainable_variables
    )

    # Replace the original function
    optimizer.apply_gradients = accumulator.apply_gradients

    return optimizer


def get_ffn_model(input_size: int, output_size: int, hidden_size: int = 64) -> keras.Model:
    inputs = layers.Input(shape=(input_size,))
    x = inputs
    x = layers.Dense(units=hidden_size, activation='tanh')(x)
    x = layers.Dense(units=hidden_size, activation='tanh')(x)
    x = layers.Dense(units=output_size, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=x)


def make_dataset(inputs, targets, batch_size: int):
    def sample_generator_():
        while True:
            idx = np.random.randint(0, len(inputs))
            yield inputs[idx].flatten(), tf.one_hot(targets[idx], depth=num_classes)

    inputs = inputs.astype(np.float32) / 255.0
    inputs = np.expand_dims(inputs, axis=-1)
    num_classes = len(set(targets))

    input_shape = (np.prod(inputs[0].shape),)
    target_shape = (num_classes,)

    return tf.data.Dataset.from_generator(
        lambda: sample_generator_(),
        output_types=(tf.float32, tf.float32),
        output_shapes=(input_shape, target_shape)
    ).padded_batch(batch_size)


def main():
    train_batch_size = 1
    valid_batch_size = 10
    grad_acc_n = 10
    steps_per_epoch = 1000 # * grad_acc_n  # Make sure we have the same number of updates

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    train_data = make_dataset(x_train, y_train, batch_size=train_batch_size)
    valid_data = make_dataset(x_test, y_test, batch_size=valid_batch_size)

    input_size = train_data.element_spec[0].shape[-1]
    output_size = train_data.element_spec[1].shape[-1]

    model = get_ffn_model(input_size=input_size, output_size=output_size, hidden_size=128)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    optimizer = get_patched_optimizer(optimizer, n=grad_acc_n, trainable_variables=model.trainable_variables)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_data,
        epochs=10,
        steps_per_epoch=steps_per_epoch // train_batch_size,
        validation_data=valid_data,
        validation_steps=10
    )


if __name__ == '__main__':
    main()

if precision_policy.name.startswith('mixed'):
    logger.info(f'Using LossScaleOptimizer (policy: "{precision_policy.name})"')
    optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)

if grad_acc_n > 1:
    # --> This patched the LossScaleOptimizer which caused the problem:
    optimizer = grad_acc.get_patched_optimizer(optimizer=optimizer, n=grad_acc_n)

if isinstance(optimizer, keras.mixed_precision.LossScaleOptimizer):
    # Warning: This does NOT work either (just an example)!
    optimizer.inner_optimizer.apply_gradients = accumulator.apply_gradients
    raise Exception('Don\'t do this!')
else:
    optimizer.apply_gradients = accumulator.apply_gradients
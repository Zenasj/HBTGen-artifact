# tf.random.uniform((B, input_size), dtype=tf.float32) ← Assumption: Input shape matches a batch of flattened MNIST images or similar

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class _GradientAccumulationPatch:
    def __init__(self, n: int, orig_apply_gradients, trainable_variables):
        self.n = tf.constant(n, dtype=tf.int64)
        policy = tf.keras.mixed_precision.global_policy()
        self.variable_dtype = policy.variable_dtype
        self.accu_gradients = [
            tf.Variable(tf.zeros(g.shape, dtype=g.dtype))
            for g in trainable_variables
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

        # Accumulate gradients using assign_add properly (fix)
        for i, grad in enumerate(gradients):
            # Guard against None gradients (e.g., non-trainable layers or zero grads)
            if grad is not None:
                self.accu_gradients[i].assign_add(grad / tf.cast(self.n, dtype=self.variable_dtype))

        # Multiply each gradient with our apply-signal (0 or 1)
        final_gradients = [grad * apply for grad in self.accu_gradients]

        # Apply the (possibly averaged) accumulated gradients to original optimizer
        self._orig_apply_gradients(zip(final_gradients, trainable_variables), *args, **kwargs)

        # Reset the accumulators when we just applied gradients (keep=0 if apply=1)
        for g in self.accu_gradients:
            g.assign(g * keep)

    def apply_accu_gradients(self, trainable_variables, *args, **kwargs):
        # Directly apply accumulated gradients via original apply_gradients
        self._orig_apply_gradients(zip(self.accu_gradients, trainable_variables), *args, **kwargs)
        # Reset all accumulated gradients to zero
        for i in range(len(self.accu_gradients)):
            self.accu_gradients[i].assign(tf.zeros_like(trainable_variables[i]))

    def _can_apply_on_next_step(self):
        """
        :return: True if gradients should be applied; False otherwise.
        This increments step count each call.
        """
        self._current_step.assign_add(1)
        count_mod_steps = tf.math.mod(self._current_step, self.n)
        return tf.equal(count_mod_steps, 0)


def get_patched_optimizer(optimizer, n, trainable_variables):
    """
    Patch an optimizer to accumulate gradients over n steps.

    The patched optimizer will accumulate gradients internally,
    and only apply the weighted average every n steps.

    Note:
      - If using mixed precision with LossScaleOptimizer,
        patching must be done carefully (not covered here).
      - This patch overrides optimizer.apply_gradients() inplace.
    """
    accumulator = _GradientAccumulationPatch(
        n=n,
        orig_apply_gradients=optimizer.apply_gradients,
        trainable_variables=trainable_variables
    )
    optimizer.apply_gradients = accumulator.apply_gradients
    return optimizer


class MyModel(tf.keras.Model):
    """
    Simple feed-forward network model adapted from MNIST example.

    Input: shape (batch_size, input_size) - flattened image or similar.
    Output: (batch_size, output_size) with softmax activation.
    """
    def __init__(self, input_size=784, output_size=10, hidden_size=128):
        super().__init__()
        self.hidden1 = layers.Dense(units=hidden_size, activation='tanh')
        self.hidden2 = layers.Dense(units=hidden_size, activation='tanh')
        self.output_layer = layers.Dense(units=output_size, activation='softmax')

    def call(self, inputs, training=None):
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        x = self.output_layer(x)
        return x


def my_model_function():
    """
    Builds and returns a MyModel instance.

    Assumes input size 784 and output size 10, suitable for MNIST-like data.

    Note: No training or compilation is done here,
    so user should compile and possibly patch optimizer externally.
    """
    return MyModel(input_size=784, output_size=10, hidden_size=128)


def GetInput():
    """
    Returns a random input tensor matching the expected input shape of MyModel.

    The input shape is (B, 784) where B is batch size.
    This uses batch size 32 and dtype float32.
    """
    batch_size = 32
    input_size = 784  # Assumed MNIST flattened size (28*28)
    return tf.random.uniform((batch_size, input_size), dtype=tf.float32)


# tf.random.uniform((B,), dtype=tf.int32) ← input is a step (scalar integer step), batch dimension not applicable here

import tensorflow as tf
from typing import List, Mapping, Any

BASE_LEARNING_RATE = 0.1  # Assuming some base learning rate for scaling


class PiecewiseConstantDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Piecewise constant decay with warmup schedule."""

    def __init__(self,
                 batch_size: int,
                 epoch_size: int,
                 warmup_epochs: int,
                 boundaries: List[int],
                 multipliers: List[float]):
        """Piecewise constant decay with warmup.

        Args:
          batch_size: The training batch size used in the experiment.
          epoch_size: The size of an epoch, or the number of examples in an epoch.
          warmup_epochs: The number of warmup epochs to apply.
          boundaries: The list of floats with strictly increasing entries.
          multipliers: The list of multipliers/learning rates to use for the
            piecewise portion. The length must be 1 less than that of boundaries.

        """
        super(PiecewiseConstantDecayWithWarmup, self).__init__()
        if len(boundaries) != len(multipliers) - 1:
            raise ValueError("The length of boundaries must be 1 less than the "
                             "length of multipliers")

        base_lr_batch_size = 256
        steps_per_epoch = epoch_size // batch_size

        self._rescaled_lr = BASE_LEARNING_RATE * batch_size / base_lr_batch_size
        # Convert epoch boundaries to step boundaries for piecewise constant decay
        self._step_boundaries = [float(steps_per_epoch) * x for x in boundaries]
        self._lr_values = [self._rescaled_lr * m for m in multipliers]
        self._warmup_steps = warmup_epochs * steps_per_epoch

    def __call__(self, step: tf.Tensor):
        """Compute learning rate at given step."""

        def warmup_lr():
            # Linearly increase learning rate from zero to base warmup LR
            return self._rescaled_lr * (
                tf.cast(step, tf.float32) / tf.cast(self._warmup_steps, tf.float32))

        def piecewise_lr():
            # Piecewise constant LR after warmup period
            return tf.compat.v1.train.piecewise_constant(
                tf.cast(step, tf.float32), self._step_boundaries, self._lr_values)

        return tf.cond(step < self._warmup_steps, warmup_lr, piecewise_lr)

    def get_config(self) -> Mapping[str, Any]:
        return {
            "rescaled_lr": self._rescaled_lr,
            "step_boundaries": self._step_boundaries,
            "lr_values": self._lr_values,
            "warmup_steps": self._warmup_steps,
        }


class MyModel(tf.keras.Model):
    """
    A Keras model wrapper to demonstrate logging of LearningRateSchedule outputs.
    
    Here, the model wraps a dummy trainable layer (e.g., dense layer) and uses a
    PiecewiseConstantDecayWithWarmup learning rate schedule. 
    
    The call method returns the current learning rate for a given training step.
    This class fuses the schedule usage and allows for easy learning rate inspection.
    """

    def __init__(self):
        super(MyModel, self).__init__()
        # Dummy layer just to have weights and a forward pass
        self.dense = tf.keras.layers.Dense(1)

        # Example parameters (these can be adjusted or made configurable)
        batch_size = 128
        epoch_size = 1000
        warmup_epochs = 5
        boundaries = [30, 60, 80]
        multipliers = [1.0, 0.1, 0.01, 0.001]

        self.lr_schedule = PiecewiseConstantDecayWithWarmup(
            batch_size=batch_size,
            epoch_size=epoch_size,
            warmup_epochs=warmup_epochs,
            boundaries=boundaries,
            multipliers=multipliers
        )

    def call(self, inputs):
        # inputs is the current step as integer tensor scalar
        x = tf.cast(inputs, tf.float32)  # cast step to float for model input
        x = tf.reshape(x, [1, 1])  # dummy shape to feed dense layer
        _ = self.dense(x)  # forward pass through dummy dense layer (not really used)

        # Compute and return current learning rate for this step
        current_lr = self.lr_schedule(inputs)
        return current_lr


def my_model_function():
    # Return an instance of MyModel, initialized with default parameters
    return MyModel()


def GetInput():
    # Return a random scalar integer step tensor as the input to the model
    # Example: a random step between 0 and 100, matching expected input to lr schedule
    return tf.random.uniform((), minval=0, maxval=100, dtype=tf.int32)


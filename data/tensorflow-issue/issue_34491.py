import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
from numpy.random import normal, randint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, Activation, Dense

time_steps = 20
num_seqs = 100
X = normal(size=(num_seqs, time_steps))  # create artificial data
Y = np.where(X > 0, 1, 0)  # create simple target

lens = randint(low=1, high=time_steps, size=num_seqs)  # create lengths < time_steps (padding needed)
seqs = [row[:row_len] for row, row_len in zip(X, lens)]  # artificially cut sequences
target_seqs = [row[:row_len] for row, row_len in zip(Y, lens)]  # artificially cut target sequences

padded_seqs = pad_sequences(seqs, padding='post', dtype='float32', maxlen=time_steps).reshape(num_seqs, time_steps, -1)
padded_targets = pad_sequences(target_seqs, dtype='float32', padding='post', maxlen=time_steps).reshape(num_seqs, time_steps, -1)

LSTM_SIZE = 16
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(time_steps,1)))
model.add(LSTM(LSTM_SIZE, return_sequences=True))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["binary_crossentropy"])
print(model.summary())

model.fit(padded_seqs, padded_targets, batch_size=1024, epochs=10)

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Masking, LSTM, Activation, Dense

from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils.losses_utils import ReductionV2, squeeze_or_expand_dimensions, _safe_mean
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.ops.ragged import ragged_tensor


def reduce_weighted_loss(weighted_losses,
                         sample_weight,  # !!!!!!!!!!!!!!!!!!!!THIS IS A CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                         reduction=ReductionV2.SUM_OVER_BATCH_SIZE):
  """Reduces the individual weighted loss measurements."""
  if reduction == ReductionV2.NONE:
    loss = weighted_losses
  else:
    loss = math_ops.reduce_sum(weighted_losses)
    if reduction == ReductionV2.SUM_OVER_BATCH_SIZE:
      loss = _safe_mean(loss, math_ops.reduce_sum(sample_weight))  # !!!!!!!!!!!!!!!!!!!!THIS IS A CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  return loss


def compute_weighted_loss(losses,
                          sample_weight=None,
                          reduction=ReductionV2.SUM_OVER_BATCH_SIZE,
                          name=None):
  """Computes the weighted loss.
  Args:
    losses: `Tensor` of shape `[batch_size, d1, ... dN]`.
    sample_weight: Optional `Tensor` whose rank is either 0, or the same rank as
      `losses`, or be broadcastable to `losses`.
    reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to loss.
      Default value is `SUM_OVER_BATCH_SIZE`.
    name: Optional name for the op.
  Raises:
    ValueError: If the shape of `sample_weight` is not compatible with `losses`.
  Returns:
    Weighted loss `Tensor` of the same type as `losses`. If `reduction` is
    `NONE`, this has the same shape as `losses`; otherwise, it is scalar.
  """
  ReductionV2.validate(reduction)

  # If this function is called directly, then we just default 'AUTO' to
  # 'SUM_OVER_BATCH_SIZE'. Eg. Canned estimator use cases.
  if reduction == ReductionV2.AUTO:
    reduction = ReductionV2.SUM_OVER_BATCH_SIZE
  if sample_weight is None:
    sample_weight = 1.0
  with backend.name_scope(name or 'weighted_loss'):
    # Save the `reduction` argument for loss normalization when distributing
    # to multiple replicas. Used only for estimator + v1 optimizer flow.
    ops.get_default_graph()._last_loss_reduction = reduction  # pylint: disable=protected-access

    if not isinstance(losses,
                      (keras_tensor.KerasTensor, ragged_tensor.RaggedTensor)):
      losses = ops.convert_to_tensor_v2_with_dispatch(losses)
    input_dtype = losses.dtype

    if not isinstance(sample_weight, keras_tensor.KerasTensor):
      sample_weight = ops.convert_to_tensor_v2_with_dispatch(sample_weight)

    # TODO(psv): Handle casting here in a better way, eg. if losses is float64
    # we do not want to lose precision.
    losses = math_ops.cast(losses, 'float32')
    sample_weight = math_ops.cast(sample_weight, 'float32')
    # Update dimensions of `sample_weight` to match with `losses` if possible.
    losses, _, sample_weight = squeeze_or_expand_dimensions(  # pylint: disable=unbalanced-tuple-unpacking
        losses, None, sample_weight)
    weighted_losses = math_ops.multiply(losses, sample_weight)

    # Apply reduction function to the individual weighted losses.
    loss = reduce_weighted_loss(weighted_losses, sample_weight, reduction)  # !!!!!!!!!!!!!!!!!!!!THIS IS A CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Convert the result back to the input type.
    loss = math_ops.cast(loss, input_dtype)
    return loss


losses_utils.compute_weighted_loss = compute_weighted_loss
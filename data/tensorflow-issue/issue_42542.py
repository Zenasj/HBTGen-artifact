# tf.random.uniform((B, None, 1), dtype=tf.float32)  # Input shape: batch size unknown, variable time steps, 1 feature

import tensorflow as tf
from collections import deque
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers

class ES(Layer):
    """
    Exponential Smoothing layer adapted for TensorFlow 2.x eager execution and graph tracing.

    Assumptions and modifications:
    - Use tf operations and tf.Tensor instead of Keras symbolic slicing to ensure Tensor compatibility.
    - seasonality queue simulated using tensors updated stepwise; here implemented as a tensor variable to avoid Python deque in call.
      However, to keep close to original logic, we simulate deque with tensor ops for TF2.
    - The horizon is assumed less than or equal to the seasonal period `m`.
    """

    def __init__(self, horizon, m, batch_size=None, time_steps=None, **kwargs):
        super(ES, self).__init__(**kwargs)
        self.horizon = horizon
        self.m = m
        self.batch_size = batch_size  # Not strictly needed but kept for legacy compatibility
        self.time_steps = time_steps

    def build(self, input_shape):
        # Learned smoothing parameters
        self.alpha = self.add_weight(
            name='alpha', shape=(1,), initializer='uniform', trainable=True
        )
        self.gamma = self.add_weight(
            name='gamma', shape=(1,), initializer='uniform', trainable=True
        )

        # Initial seasonality vector of shape (m,)
        self.init_seasonality = self.add_weight(
            name='init_seasonality',
            shape=(self.m,),
            initializer=initializers.Constant(value=0.8),
            trainable=True,
        )

        # Initial level scalar
        self.level = self.add_weight(
            name='init_level',
            shape=(1,),
            initializer=initializers.Constant(value=0.8),
            trainable=True,
        )

        super(ES, self).build(input_shape)

    def call(self, x):
        """
        Args:
          x: Input tensor of shape (batch_size, time_steps, 1).

        Returns:
          A list [x_out, denorm_coeff] where:
            x_out: normalized features tensor of shape (batch_size, horizon, 1)
            denorm_coeff: tensor of shape (batch_size, horizon, 2)
              where denorm_coeff[:,:,0] = levels repeated,
                    denorm_coeff[:,:,1] = seasonality terms
        """
        # Get batch and time dimensions dynamically
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]

        # We assume input has last dim 1
        x = tf.reshape(x, (batch_size, time_steps))  # shape (B, T)

        # Initialize seasonality "queue" as a tensor of shape (m,)
        seasonality = tf.identity(self.init_seasonality)  # shape (m,)

        level = self.level[0]  # scalar

        # Prepare lists to collect normalized values and level-seasonality composites
        x_norm_list = []
        ls_list = []

        # Flatten all series in batch & time dimension as a single sequence for ES computation
        # The original code concatenates a slice from first example and then others.
        #
        # This behavior:
        #   ts = concat([x[0,0:time_steps], x[1:, -1]])
        # was used but is a bit convoluted. Here we process each batch time series independently for clarity.

        # To respect original logic: extract ts as concatenation of first example full time series and
        #   one time-step from all subsequent series:
        # But for simplicity with tf.function, we just flatten batch/time dimension:
        ts = tf.reshape(x, [-1])  # shape (batch_size * time_steps,)

        total_steps = tf.shape(ts)[0]

        # Looping with tf.while_loop because Python loops with deque are not traceable.
        # We implement deque with a tensor slice updated each step.

        def cond(i, level, seasonality, x_norm_list, ls_list):
            return i < total_steps

        def body(i, level, seasonality, x_norm_list, ls_list):
            y_t = ts[i]

            # seasonality index (mod m)
            idx = tf.math.floormod(i, self.m)
            s_t = seasonality[idx]

            # level update
            l_t = self.alpha * y_t / s_t + (1.0 - self.alpha) * level

            # seasonality update
            s_t_plus_m = self.gamma * y_t / l_t + (1.0 - self.gamma) * s_t

            # Update seasonality vector at idx
            seasonality = tf.tensor_scatter_nd_update(seasonality, [[idx]], [s_t_plus_m])

            # normalized value
            x_norm = y_t / (s_t * l_t)
            x_norm_list = x_norm_list.write(i, x_norm)

            # collect denorm coeff after time_steps-1 (when forecasting starts)
            def add_ls():
                # repeat level and seasonality for horizon
                l_repeat = tf.repeat(l_t, repeats=self.horizon)  # (horizon,)
                # seasonality slice for horizon starting from idx+1 mod m
                # Since horizon < m, we can create slice safely
                season_idx = tf.range(idx + 1, idx + 1 + self.horizon) % self.m
                s_repeat = tf.gather(seasonality, season_idx)  # (horizon,)
                # stack level and seasonality as (horizon, 2)
                ls_t = tf.stack([l_repeat, s_repeat], axis=1)
                ls_list = ls_list.write(i - (self.time_steps - 1), ls_t)
                return ls_list

            ls_list = tf.cond(
                i >= (self.time_steps - 1),
                add_ls,
                lambda: ls_list
            )

            return i + 1, l_t, seasonality, x_norm_list, ls_list

        # Using tf.TensorArray to collect values
        x_norm_array = tf.TensorArray(dtype=tf.float32, size=total_steps)
        ls_array = tf.TensorArray(dtype=tf.float32, size=total_steps - self.time_steps + 1)

        i0 = tf.constant(0)
        i, level_final, seasonality_final, x_norm_array, ls_array = tf.while_loop(
            cond,
            body,
            loop_vars=[i0, level, seasonality, x_norm_array, ls_array],
            shape_invariants=[
                i0.get_shape(),
                level.get_shape(),
                seasonality.get_shape(),
                tf.TensorShape(None),
                tf.TensorShape(None),
            ],
            maximum_iterations=total_steps,
        )

        # x_norm: shape (total_steps,)
        x_norm = x_norm_array.stack()

        # ls: (num_forecasts, horizon, 2)
        num_forecasts = total_steps - self.time_steps + 1
        ls = ls_array.stack()

        # Now split normalized values back into batch/time slices for output
        # The original method creates a tensor (batch_size, horizon, 1)
        # But here horizon is fixed, so we extract x_norm segments for each example

        # Since we do not have the exact logic how to reshape x_norm,
        # we produce normalized features as (batch_size, self.time_steps)
        x_norm_reshaped = tf.reshape(x_norm[: batch_size * self.time_steps], (batch_size, self.time_steps))

        # We expand dims consistent with original code (adds axis=2)
        x_out = tf.expand_dims(x_norm_reshaped, axis=2)  # (batch_size, time_steps, 1)

        # The denorm_coeff ls has shape (num_forecasts, horizon, 2)
        # We try to reshape/pad to return shape (batch_size, horizon, 2) for downstream processing.
        # Since num_forecasts = total_steps - time_steps +1 = batch_size * time_steps - time_steps +1,
        # to match batch_size, pick the first batch_size entries assuming horizon < m holds.

        denorm_coeff = tf.cond(
            num_forecasts >= batch_size,
            lambda: ls[:batch_size, :, :],  # take first batch_size forecasts
            lambda: tf.repeat(ls[0:1, :, :], repeats=batch_size, axis=0)  # fallback repeat first
        )

        return [x_out, denorm_coeff]

    def compute_output_shape(self, input_shape):
        # Returns two tensors shapes:
        # normalized time series features: (batch_size, time_steps, 1)
        # denorm coefficients: (batch_size, horizon, 2)
        return [
            (input_shape[0], input_shape[1], 1),
            (input_shape[0], self.horizon, 2),
        ]


class Denormalization(Layer):
    """
    Layer to denormalize output:
    output = normalized * level * seasonality
    """

    def call(self, x):
        # x is a list of two tensors: [normalized_values, denorm_coeff]
        normalized = x[0]  # shape (B, time_steps, 1)
        denorm_coeff = x[1]  # shape (B, horizon, 2)

        # We assume the second dimension matches horizon (maybe smaller than time_steps)
        # So we slice normalized to horizon time steps, squeeze last dim and multiply by level * seasonality

        # Crop normalized to horizon size
        normalized_horizon = normalized[:, : tf.shape(denorm_coeff)[1], 0]  # shape (B, horizon)

        # level and seasonality from denorm_coeff
        level = denorm_coeff[:, :, 0]  # shape (B, horizon)
        seasonality = denorm_coeff[:, :, 1]  # shape (B, horizon)

        output = normalized_horizon * level * seasonality  # shape (B, horizon)

        return output

    def compute_output_shape(self, input_shape):
        # Returns shape: (batch_size, horizon)
        normalized_shape, denorm_shape = input_shape
        return (normalized_shape[0], denorm_shape[1])


class MyModel(tf.keras.Model):
    """
    Composite model fusing ES and Denormalization layers.
    Forward pass outputs denormalized forecasts.
    """

    def __init__(self, horizon=3, m=26, time_steps=6):
        super(MyModel, self).__init__()
        self.es = ES(horizon=horizon, m=m, time_steps=time_steps)
        self.gru = tf.keras.layers.GRU(units=5)
        self.dense = tf.keras.layers.Dense(units=horizon)
        self.denorm = Denormalization()

    def call(self, inputs):
        """
        Args:
          inputs: tensor of shape (batch_size, time_steps, 1)

        Returns:
          tensor of denormalized forecasts, shape (batch_size, horizon)
        """

        # ES layer returns normalized features and denorm coeffs
        x_out, denorm_coeff = self.es(inputs)  # shapes (B, time_steps, 1), (B, horizon, 2)

        # Pass normalized features through GRU and Dense
        gru_out = self.gru(x_out)  # (B, units)
        dense_out = self.dense(gru_out)  # (B, horizon)

        # The dense outputs normalized forecasts for horizon (batch_size, horizon),
        # but ES.denorm expects [normalized, denorm_coeff]
        # We need to expand dims to shape (B, horizon, 1)
        normalized_forecast = tf.expand_dims(dense_out, axis=2)

        # Denormalize forecasts
        out = self.denorm([normalized_forecast, denorm_coeff])  # shape (B, horizon)

        return out


def my_model_function():
    # Instantiate MyModel with default parameters
    return MyModel()


def GetInput():
    # Generate a random tensor input matching expected input shape for MyModel call
    # Input shape: (batch_size, time_steps, 1)
    # Using defaults: batch_size=48, time_steps=6 (from original example)
    batch_size = 48
    time_steps = 6
    return tf.random.uniform((batch_size, time_steps, 1), dtype=tf.float32)


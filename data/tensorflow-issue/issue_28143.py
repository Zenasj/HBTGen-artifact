# tf.random.uniform((B, T, F), dtype=tf.float32) ← Input shape assumed as (batch_size, time_steps, features)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Masking layer masks inputs equal to 0.0
        self.masking = tf.keras.layers.Masking(mask_value=0.0)
        # Lambda layer applying sigmoid activation
        # Note: By default, Lambda layer does not support masking, and silently ignores input mask.
        self.lambda_layer = tf.keras.layers.Lambda(tf.nn.sigmoid)
        # GlobalMaxPool1D layer that does NOT support masking and raises error if an input mask is passed
        self.global_max_pool = tf.keras.layers.GlobalMaxPool1D()

    def call(self, inputs, training=None, mask=None):
        # Apply masking layer first
        x = self.masking(inputs)

        # Check mask passed from masking layer
        input_mask = self.masking.compute_mask(inputs)

        # Apply lambda_layer, which ignores masking by default
        x_lambda = self.lambda_layer(x, mask=input_mask)

        # Apply global max pooling, which raises if mask is passed
        # Because GlobalMaxPool1D doesn't support masking, we must ensure no mask is passed,
        # Otherwise TensorFlow raises TypeError.
        # To emulate that, we explicitly pass no mask to this layer.

        # If input_mask is not None, passing it to global_max_pool would error out.
        # So drop mask before passing to global_max_pool.
        # This models the behavior discussed in the issue.

        x_pool = self.global_max_pool(x)  # no mask argument allowed

        # Compare outputs: shapes may differ (lambda output preserves timesteps, pool output reduces)
        # Here, to comply with the "fuse models and implement comparison",
        # we compare lambda output pooled vs global max pooled output by reducing lambda output.

        # For comparison, reduce lambda output max over time axis (axis=1)
        lambda_pooled = tf.reduce_max(x_lambda, axis=1)

        # Comparison: check if pooled outputs are close elementwise within tol
        tol = 1e-5
        comparison = tf.math.abs(lambda_pooled - x_pool) < tol

        # Return dict of outputs and boolean comparison
        # Following the fusion requirement: output reflects comparison (boolean tensor)
        return {
            "lambda_pooled": lambda_pooled,
            "global_max_pool": x_pool,
            "are_close": comparison
        }

def my_model_function():
    # Return an instance of MyModel; no special initialization needed
    return MyModel()

def GetInput():
    # Return a random float32 tensor of shape (batch_size=2, timesteps=5, features=3),
    # matching the input expected by MyModel.
    # Values randomly sampled from uniform [0,1).
    return tf.random.uniform((2, 5, 3), dtype=tf.float32)


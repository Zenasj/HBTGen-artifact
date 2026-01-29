# tf.random.uniform((B=160, H=32, W=32, C=3), dtype=tf.float32)

import tensorflow as tf
from tensorflow import keras

class NormDense(keras.layers.Layer):
    def __init__(self, units=10, append_norm=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.append_norm = append_norm

    def build(self, input_shape):
        # Weight shape: [input_features, units]
        self.w = self.add_weight(
            name="norm_dense_w",
            shape=(input_shape[-1], self.units),
            trainable=True,
            initializer="glorot_uniform",
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # Normalize weights over axis 0 (columns)
        norm_w = tf.nn.l2_normalize(self.w, axis=0)
        # Normalize inputs over axis 1 (features per example)
        norm_inputs = tf.nn.l2_normalize(inputs, axis=1)
        output = tf.matmul(norm_inputs, norm_w)
        if self.append_norm:
            # Append negative norm of inputs as an extra feature
            norm_feature = tf.norm(inputs, axis=1, keepdims=True) * -1
            output = tf.concat([output, norm_feature], axis=-1)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "append_norm": self.append_norm})
        return config


class NormDenseLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        # y_pred shape could be [batch_size, units] or [batch_size, units+1] if append_norm=True
        if y_pred.shape[-1] == y_true.shape[-1]:
            # Normal case: no appended norm feature
            norm_logits = y_pred
            margin = 0.3
            regularizer_loss = 0.0
        else:
            # When append_norm=True, last channel is negative norm feature
            norm_logits = y_pred[:, :-1]
            feature_norm = y_pred[:, -1] * -1
            # Margin depends on feature_norm and introduces XLA issue in TF 2.8+
            margin = 0.04 * (feature_norm - 10) + 10.0
            regularizer_loss = feature_norm / 1e4 + 1.0 / feature_norm

        # Identify indices where y_true > 0 (positive class locations)
        pick_cond = tf.where(y_true > 0)
        # Gather predicted logits at those indices
        y_pred_vals = tf.gather_nd(norm_logits, pick_cond)
        theta_valid = y_pred_vals - margin

        # Update logits at picked positions with theta_valid
        logits = tf.tensor_scatter_nd_update(norm_logits, pick_cond, theta_valid)

        # Compute categorical crossentropy loss with updated logits
        cls_loss = tf.keras.losses.categorical_crossentropy(
            y_true, logits, from_logits=self.from_logits
        )

        # Add regularization weighted term
        total_loss = cls_loss + regularizer_loss * 35.0
        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({"from_logits": self.from_logits})
        return config


class MyModel(tf.keras.Model):
    def __init__(self, units=10, append_norm=False):
        super().__init__()
        self.append_norm = append_norm
        # Model: Flatten input, Dense 32, then NormDense with units and append_norm
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(32, activation=None)
        self.norm_dense = NormDense(units=units, append_norm=append_norm)

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense(x)
        x = self.norm_dense(x)
        return x


def my_model_function():
    # Returns MyModel instance with append_norm=True to reproduce the original scenario
    # (append_norm=True triggers the margin-based loss path)
    return MyModel(units=10, append_norm=True)


def GetInput():
    # Generate a random input tensor of shape [160, 32, 32, 3]
    # dtype float32 to match the expected model input
    return tf.random.uniform((160, 32, 32, 3), dtype=tf.float32)


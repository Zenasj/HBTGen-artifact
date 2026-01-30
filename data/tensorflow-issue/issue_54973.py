import random
from tensorflow.keras import layers
from tensorflow.keras import models

py
#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras


class NormDense(keras.layers.Layer):
    def __init__(self, units=1000, append_norm=False, **kwargs):
        super().__init__(**kwargs)
        self.units, self.append_norm = units, append_norm

    def build(self, input_shape):
        self.w = self.add_weight(name="norm_dense_w", shape=(input_shape[-1], self.units), trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # tf.print("tf.reduce_mean(self.w):", tf.reduce_mean(self.w))
        norm_w = tf.nn.l2_normalize(self.w, axis=0)
        norm_inputs = tf.nn.l2_normalize(inputs, axis=1)
        output = tf.matmul(norm_inputs, norm_w)
        if self.append_norm:
            output = tf.concat([output, tf.norm(inputs, axis=1, keepdims=True) * -1], axis=-1)
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
        if y_pred.shape[-1] == y_true.shape[-1]:
            norm_logits = y_pred
            margin = 0.3
            regularizer_loss = 0.0
        else:
            norm_logits, feature_norm = y_pred[:, :-1], y_pred[:, -1] * -1
            margin = 0.04 * (feature_norm - 10) + 10.0  # This triggers the error
            regularizer_loss = feature_norm / 1e4 + 1.0 / feature_norm

        pick_cond = tf.where(y_true > 0)
        y_pred_vals = tf.gather_nd(norm_logits, pick_cond)
        theta_valid = y_pred_vals - margin

        # tf.print(">>>>", norm_logits.shape, pick_cond, tf.reduce_sum(tf.cast(y_true > 0, "float32")), theta_valid.shape)
        logits = tf.tensor_scatter_nd_update(norm_logits, pick_cond, theta_valid)
        # theta_one_hot = tf.expand_dims(theta_valid, 1) * tf.cast(y_true, dtype=tf.float32)
        # logits = tf.where(tf.cast(y_true, dtype=tf.bool), theta_one_hot, norm_logits)
        # tf.print(">>>>", norm_logits.shape, logits.shape, y_true.shape)
        cls_loss = tf.keras.losses.categorical_crossentropy(y_true, logits, from_logits=self.from_logits)

        # tf.print(">>>>", cls_loss.shape, regularizer_loss.shape)
        return cls_loss + regularizer_loss * 35.0

    def get_config(self):
        config = super().get_config()
        config.update({"from_logits": self.from_logits})
        return config


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--append_norm", action="store_true", help="append norm")
    args = parser.parse_known_args(sys.argv[1:])[0]

    xx = tf.random.uniform([160, 32, 32, 3])
    yy = tf.one_hot(tf.cast(tf.random.uniform([160], 0, 10), "int32"), 10)
    mm = keras.models.Sequential([keras.layers.Input([32, 32, 3]), keras.layers.Flatten(), keras.layers.Dense(32), NormDense(10, append_norm=args.append_norm)])
    mm.compile(loss=NormDenseLoss(), optimizer="adam")
    mm.fit(xx, yy)
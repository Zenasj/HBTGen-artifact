# tf.random.uniform((B, 1), dtype=tf.float32) ← original input shape is (batch_size, 1)

import tensorflow as tf
from tensorflow.keras import backend as K

class MyModel(tf.keras.Model):
    def __init__(self, rate_mse=1e5, rate_reg=5e-2):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(20, activation='relu')
        self.reg_rate = rate_reg
        self.output_layer = tf.keras.layers.Dense(1, activation='linear')
        self.mse_rate = rate_mse

    # Custom MSE loss function returning per-sample losses (vector)
    def rate_mse_loss(self, y_true, y_pred):
        # mean squared error per sample averaged over last axis
        # scaled with mse_rate
        loss_per_sample = self.mse_rate * K.mean(K.square(y_pred - y_true), axis=-1)
        return loss_per_sample  # shape: (batch_size,)

    def call(self, inputs, training=None):
        x = self.dense1(inputs)  # (batch_size, 20)
        # The regularizer loss must be scalar per batch sample
        # The issue from the original code was that axis was -1 which only reduces last dim,
        # leaving shape (batch_size,), which is compatible as scalar loss per sample.
        # But Keras requires losses to be scalar tensor, so we add scalar loss with add_loss per batch step.

        # Compute regularization loss as mean squared activation per sample:
        # (batch_size,)
        reg_loss_per_sample = self.reg_rate * K.mean(x * x, axis=-1)

        # Add scalar loss to model losses (Keras expects scalar loss tensors, so reduce batch axis)
        reg_loss_scalar = K.mean(reg_loss_per_sample)
        self.add_loss(reg_loss_scalar, inputs=True)

        output = self.output_layer(x)
        return output

    def compute_loss(self, y_true, y_pred):
        # Return scalar loss for compiled loss function in Keras:
        # reduce rate_mse_loss vector to scalar to be compatible with model-level loss aggregation
        mse_vector = self.rate_mse_loss(y_true, y_pred)
        return K.mean(mse_vector)

def my_model_function():
    # Return an instance of MyModel initialized as in the shared code.
    return MyModel()

def GetInput():
    # From code: input shape is (batch_size, 1)
    # batch size used in example is 100, but can be any integer; here let's pick 100 for consistency.
    batch_size = 100
    return tf.random.uniform((batch_size, 1), dtype=tf.float32)


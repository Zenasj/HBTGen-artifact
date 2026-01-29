# tf.random.uniform((B, 500, 5), dtype=tf.float32) ← inferred input shape from example: (batch, timesteps=500, features=5)

import tensorflow as tf
from tensorflow.keras import layers, backend as K

class Masked_MAE(tf.keras.metrics.Metric):
    def __init__(self, name='masked_mae', mean=0, std=1.0, **kwargs):
        super(Masked_MAE, self).__init__(name=name, **kwargs)
        # Store scaling factor
        self.mean = mean
        self.std = std
        self.factor = self.std
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            # Create mask: 1.0 where sample_weight!=0, else 0.0
            mask = K.cast(K.not_equal(sample_weight, 0), K.floatx())
            # Expand mask dims to broadcast over y_true last dimension
            mask = tf.expand_dims(mask, axis=-1)
            multp = tf.cast(tf.shape(y_true)[-1], dtype=K.floatx())
        else:
            mask = tf.ones_like(y_true, dtype=K.floatx())
            multp = 1.0
        masked_error = K.abs(y_true - y_pred) * mask
        self.total.assign_add(K.sum(masked_error))
        self.count.assign_add(K.sum(mask) * multp)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count) * self.factor

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define layers similar to the original example
        self.lstm = layers.LSTM(30, return_sequences=True)
        self.dense1 = layers.Dense(1, name='dense_output1')
        self.dense2 = layers.Dense(1, name='dense_output2')

        # Instantiate custom masked MAE metric for both outputs
        self.masked_mae_1 = Masked_MAE(name='masked_mae_1')
        self.masked_mae_2 = Masked_MAE(name='masked_mae_2')

    def call(self, inputs, training=False):
        x = self.lstm(inputs)
        out1 = self.dense1(x)
        out2 = self.dense2(x)
        return out1, out2

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor of shape (batch=2, 500, 5) with dtype float32
    # The batch size 2 is arbitrary but must be >0; 2 chosen to demonstrate batching
    return tf.random.uniform((2, 500, 5), dtype=tf.float32)


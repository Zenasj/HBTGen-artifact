# tf.random.uniform((B, 2), dtype=tf.float32) ← inferred input shape is (batch_size, 2)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(4)
        self.dense2 = tf.keras.layers.Dense(4)
        # Using MeanSquaredError loss instances as attributes to emulate multiple losses
        self.loss_fn1 = tf.keras.losses.MeanSquaredError()
        self.loss_fn2 = tf.keras.losses.MeanSquaredError()

    def call(self, inputs, training=False):
        o1 = self.dense1(inputs)
        o2 = self.dense2(inputs)
        return [o1, o2]

    def compute_losses(self, y_true, y_pred):
        # y_true and y_pred are expected to be lists/tuples of two tensors each
        loss1 = self.loss_fn1(y_true[0], y_pred[0])
        loss2 = self.loss_fn2(y_true[1], y_pred[1])
        return loss1, loss2

def my_model_function():
    return MyModel()

def GetInput():
    # Return a batch of inputs with shape (batch_size=5, features=2) matching the model's input
    # Use uniform float32 tensors that simulate realistic input values
    return tf.random.uniform((5, 2), dtype=tf.float32)


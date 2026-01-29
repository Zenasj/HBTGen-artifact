# tf.random.uniform((Batch, Features), dtype=tf.float32) ← Assuming input shape for regression example is (None, 10) where 10 is feature count

import tensorflow as tf
from tensorflow import keras


class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    @tf.function
    def call(self, y_true, y_pred):
        error = tf.abs(y_true - y_pred)
        is_small_error = error <= self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = error * self.threshold - 0.5 * self.threshold**2
        return tf.where(is_small_error, squared_loss, linear_loss)

    def get_config(self):
        cfg = super().get_config()
        cfg['threshold'] = self.threshold
        return cfg


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Assumed input shape with 10 features (can adapt as needed)
        self.dense1 = keras.layers.Dense(30, activation='relu')
        self.dense2 = keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        output = self.dense2(x)
        return output


def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile with the custom HuberLoss with threshold 2.0 as example
    model.compile(loss=HuberLoss(threshold=2.0), optimizer="sgd")
    return model


def GetInput():
    # Generate a random input tensor consistent with model input (batch size 32, 10 features)
    # dtype float32 to match typical regression inputs
    return tf.random.uniform((32, 10), dtype=tf.float32)


# tf.random.uniform((B, D), dtype=tf.float32)  # Assuming input shape (batch_size, features)

import tensorflow as tf
from tensorflow import keras

class MyModel(tf.keras.Model):
    # This model demonstrates usage with a custom HuberLoss as shown in the issue.
    # The model structure is a simple 2-layer dense network for regression.
    
    def __init__(self):
        super().__init__()
        # Based on the example, first Dense with 30 units + relu, then output unit 1
        self.dense1 = keras.layers.Dense(30, activation='relu')
        self.dense2 = keras.layers.Dense(1)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)


class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
    
    @tf.function
    def call(self, y_true, y_pred):
        error = tf.abs(y_true - y_pred)
        is_small_error = error <= self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = error * self.threshold - 0.5 * self.threshold ** 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    
    def get_config(self):
        cfg = super().get_config()
        cfg['threshold'] = self.threshold
        return cfg


def my_model_function():
    # Return an instance of MyModel, including the custom loss as a member attribute
    # so it can be accessed if needed. (Not part of typical keras.Model API but 
    # shown here just for completeness from the context)
    model = MyModel()
    # We do not compile here because the requirement was only to return the model
    # and not the compiling/training code.
    return model


def GetInput():
    # Create a random tensor matching the example input shape.
    # The example uses "input_shape=X_train.shape[1:]".
    # We infer it to be a 2D float tensor of shape (batch_size, features).
    # Let's assume batch_size=32, feature dimension = 10, a generic reasonable choice.
    # dtype float32 is standard for Keras models.
    batch_size = 32
    feature_dim = 10
    return tf.random.uniform((batch_size, feature_dim), dtype=tf.float32)


# tf.random.uniform((B, 10), dtype=tf.float32) ← inferred input shape from Input(shape=(10,))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define Dense layers as used in the example
        self.dense1 = tf.keras.layers.Dense(512)
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.dense3 = tf.keras.layers.Dense(128, activation='relu')
        self.prediction_dense = tf.keras.layers.Dense(1)
        self.concat = tf.keras.layers.Concatenate()

    def call(self, inputs):
        all_layers = []

        x1 = self.dense1(inputs)
        all_layers.append(x1)

        x2 = self.dense2(x1)
        all_layers.append(x2)

        # Concatenate layers: the issue in the original was about passing list vs non-list.
        # Here we always pass a list to avoid the error.
        conc1 = self.concat(list(all_layers))
        x3 = self.dense3(conc1)
        all_layers.append(x3)

        conc2 = self.concat(list(all_layers))
        prediction = self.prediction_dense(conc2)

        return prediction

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return random input tensor with shape (batch_size=1, 10)
    # dtype float32 as typical for Dense inputs
    return tf.random.uniform((1, 10), dtype=tf.float32)


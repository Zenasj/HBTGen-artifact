# tf.random.uniform((10, 1), dtype=tf.float32) ← inferred input shape from original example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstruct the original Sequential model described:
        # Two Dense layers, first with input shape matching (1,)
        self.dense1 = tf.keras.layers.Dense(1)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Since the original example compiles with Adam and MAE loss,
    # we replicate compilation just for completeness.
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.MeanAbsoluteError())
    return model

def GetInput():
    # Return a random tensor input shaped (10,1) as in the original example
    # Use tf.float32 type (standard for TF models)
    return tf.random.uniform((10, 1), dtype=tf.float32)


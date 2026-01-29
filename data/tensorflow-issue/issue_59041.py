# tf.random.uniform((B=5, H=10, W=50), dtype=tf.float32)

import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class Custom_Layer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(Custom_Layer, self).__init__(**kwargs)
        self.units = units
        # Create a Dense layer here to avoid recreating in call
        self.dense = tf.keras.layers.Dense(units)

    def call(self, x):
        return self.dense(x)

    def get_config(self):
        config = super(Custom_Layer, self).get_config()
        config.update(units=self.units)
        return config

class MyModel(tf.keras.Model):
    def __init__(self, units=256, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        # Our custom layer that caused the issue
        self.custom_layer = Custom_Layer(units)
        # Final dense layer outputting 1 unit per time step
        self.final_dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.custom_layer(inputs)  # shape: (batch, 10, units)
        x = self.final_dense(x)        # shape: (batch, 10, 1)
        # The original code returned x[:,-1,:] but squeezed it:
        # Here we squeeze the last two dims to shape (batch,)
        output = tf.squeeze(x[:, -1, :], axis=-1)
        return output

def my_model_function():
    # Return an instance of MyModel with units=256 as used in the example
    return MyModel(units=256)

def GetInput():
    # Input shape in the example: (batch, 10, 50)
    # Use batch size 5 as in the test prediction example
    return tf.random.uniform(shape=(5, 10, 50), dtype=tf.float32)


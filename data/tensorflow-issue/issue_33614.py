# tf.random.uniform((B, ..., ...), dtype=tf.float32) ← Input shape is inferred as (batch_size, features) since keras.layers.Dense expects (..., features)

import tensorflow as tf

class SerializableNestedInt(int):
    """A serializable object containing a serializable function."""
    def __new__(cls, value, fn):
        obj = int.__new__(cls, value)
        obj.fn = fn
        return obj

    def get_config(self):
        return {'value': int(self), 'fn': self.fn}

    @classmethod
    def from_config(cls, config):
        # Deserialize using stored config dictionary
        return cls(**config)


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A serializable function to be used inside SerializableNestedInt and tested for correct serialization
        def serializable_fn(x):
            """A serializable function to pass out of a test layer's config."""
            return x

        self.serializable_fn = serializable_fn
        # Units set as SerializableNestedInt, demonstrating a nested serializable config object
        self.units_val = SerializableNestedInt(3, serializable_fn)

        # Creating a Dense layer with:
        # - units: SerializableNestedInt(3, serializable_fn) -- custom serializable int type
        # - name: 'SerializableNestedInt' (matches the class name) to illustrate the serialization challenge described in the issue
        # - activation: relu
        # - kernel_initializer: ones
        # - bias_regularizer: l2
        self.dense = tf.keras.layers.Dense(
            units=self.units_val,
            name='SerializableNestedInt',
            activation='relu',
            kernel_initializer='ones',
            bias_regularizer='l2'
        )

    def call(self, inputs):
        # Forward pass through the dense layer
        return self.dense(inputs)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Since Dense expects rank-2 input shape (batch_size, features), we choose:
    # batch_size = 4 (arbitrary)
    # features = 5 (arbitrary)
    return tf.random.uniform((4, 5), dtype=tf.float32)


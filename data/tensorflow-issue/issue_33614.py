from tensorflow.keras import layers

import tensorflow as tf
import tensorflow.keras as keras


class SerDeTest(tf.test.TestCase):
    def test_nested_serializable_fn(self):
        def serializable_fn(x):
            """A serializable function to pass out of a test layer's config."""
            return x

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
                return cls(**config)

        layer = keras.layers.Dense(
            SerializableNestedInt(3, serializable_fn),
            name='SerializableNestedInt',
            activation='relu',
            kernel_initializer='ones',
            bias_regularizer='l2')
        config = keras.layers.serialize(layer)
        new_layer = keras.layers.deserialize(
            config,
            custom_objects={
                'serializable_fn': serializable_fn,
                'SerializableNestedInt': SerializableNestedInt
            })
        self.assertEqual(new_layer.activation, keras.activations.relu)
        self.assertIsInstance(new_layer.bias_regularizer, keras.regularizers.L1L2)
        self.assertIsInstance(new_layer.units, SerializableNestedInt)
        self.assertEqual(new_layer.units, 3)
        self.assertIs(new_layer.units.fn, serializable_fn)


if __name__ == "__main__":
    tf.test.main()
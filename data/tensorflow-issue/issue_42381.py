# tf.random.uniform((B, 5), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Initialize 'a' as a non-trainable tf.Variable with shape [5].
        # Important: Assign with zeros initially to allow restoring weights from .h5 properly.
        # Later it can be set externally.
        self.a = tf.Variable(initial_value=tf.zeros([5]), trainable=False, name='a')

        # Compile methods with input_signature to avoid retracing.
        # Using input signature with shape [None, 5] to represent batch size unknown but last dim = 5
        self.call = tf.function(func=self.call, input_signature=[
            tf.TensorSpec(shape=[None, 5], dtype=tf.float32)
        ])
        self.prod = tf.function(func=self.prod, input_signature=[
            tf.TensorSpec(shape=[None, 5], dtype=tf.float32)
        ])
        self.set = tf.function(func=self.set, input_signature=[
            tf.TensorSpec(shape=[5], dtype=tf.float32)
        ])

        # Build the model to set up weights & track variables properly
        self.build(input_shape=tf.TensorShape([None, 5]))

    @tf.function
    def call(self, inputs):
        # Return element-wise addition of self.a and inputs
        return self.a + inputs

    @tf.function
    def prod(self, inputs):
        # Return element-wise product of self.a and inputs
        return self.a * inputs

    @tf.function
    def set(self, value):
        # Assign a new value to self.a variable
        self.a.assign(value)

    def get_config(self):
        # Required for saving/loading in 'h5' format when subclassing keras.Model.
        # Note: tf.Variable 'a' is converted to a tensor for config serialization.
        config = super(MyModel, self).get_config()
        config.update({"a": self.a.numpy()})
        return config

    @classmethod
    def from_config(cls, config):
        # Provide from_config so model can be reconstructed from config
        a_value = config.pop("a")
        model = cls()
        model.a.assign(a_value)
        return model


def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()


def GetInput():
    # Return a random tensor input matching [batch_size, 5] shape expected by MyModel
    # Using batch size = 2 here consistent with the example usage in the issue
    return tf.random.uniform((2, 5), dtype=tf.float32)


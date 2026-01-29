# tf.random.uniform((B, 1), dtype=tf.float32) ← inferred input shape from example Input(shape=(1,))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original issue describes behavior differences between using a
        # tf op (e.g. tf.identity) directly in Keras functional API vs using
        # a Keras layer like Lambda or Dense for naming.
        #
        # To faithfully capture the explanation and best practice, this model
        # layer wraps a tf.identity op inside a Lambda layer with a name
        # 'test_layer' which is the recommended way to explicitly name layers
        # when using tf ops in Keras 2.4+.
        #
        # This reflects the recommended approach to have stable layer naming,
        # avoiding the legacy issue of tf op layers losing the custom name.

        self.lambda_identity = tf.keras.layers.Lambda(lambda x: tf.identity(x), name='test_layer')

    def call(self, inputs):
        return self.lambda_identity(inputs)


def my_model_function():
    # Return an instance of MyModel with the Lambda identity layer named 'test_layer'
    return MyModel()


def GetInput():
    # The expected input shape is (batch_size, 1) based on the examples with Input(shape=(1,))
    # Use float32 uniform random tensor for typical input
    batch_size = 4  # example batch size; arbitrary positive number
    input_shape = (batch_size, 1)
    return tf.random.uniform(input_shape, dtype=tf.float32)


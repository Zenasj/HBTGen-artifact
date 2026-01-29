# tf.random.uniform((B, 1), dtype=tf.float32) ← Input shape inferred from Input layer batch_shape=(None, 1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Example is a dynamic custom layer that outputs a shape (None, 2) regardless of input shape (None, 1)
        # It simply returns the inputs in call, but compute_output_shape forces output shape to (None, 2)
        class Example(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                kwargs["dynamic"] = True
                super(Example, self).__init__(**kwargs)

            def call(self, inputs):
                # No change: just forward input
                return inputs

            def compute_output_shape(self, input_shape):
                # Important: Must return a tf.TensorShape to avoid the issue described in the original bug report
                return tf.TensorShape([None, 2])

        self.example_layer = Example()

    def call(self, inputs):
        return self.example_layer(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor with shape (batch_size, 1) matching Input layer's batch_shape of (None, 1)
    # Using float32 as typical default dtype for Keras inputs
    batch_size = 4  # arbitrary non-None batch size for testing
    return tf.random.uniform((batch_size, 1), dtype=tf.float32)


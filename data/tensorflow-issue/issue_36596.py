# tf.random.uniform((B, H, W, C), dtype=tf.float32)  # inferred shape from example: (batch_size, 180, 320, 3)

import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # We instantiate a simple MobileNet base (like in the example)
        # with input shape (180, 320, 3) and no pretrained weights.
        self.base_model = tf.keras.applications.MobileNet(
            input_shape=(180, 320, 3), weights=None, include_top=False
        )
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        # Forward pass through MobileNet base
        x = self.base_model(inputs, training=training)
        x = self.gap(x)
        x = self.dense(x)
        return x

    def compute_input_gradient(self, inputs, targets, loss_fn):
        """
        Compute gradients of loss_fn(targets, model(inputs)) w.r.t inputs.
        Assumes inputs is a tf.Variable or watched tf.Tensor.
        """

        with tf.GradientTape() as tape:
            tape.watch(inputs)  # ensure the input is watched because it's not a Variable by default
            preds = self.call(inputs, training=False)
            loss = loss_fn(targets, preds)

        grads = tape.gradient(loss, inputs)
        return grads


def my_model_function():
    # Create and return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input that matches MyModel's input shape,
    # batch size 1 here for simplicity.
    # dtype float32 as MobileNet expects that dtype.
    return tf.random.uniform((1, 180, 320, 3), dtype=tf.float32)


# Additional notes:
# The core issue described in the GitHub issue is that tape.gradient(loss, model.input)
# returns None because model.input is a symbolic tensor, not a watched variable/tensor.
# The remedy is to ensure the actual input tensor passed to the model is a watched tensor
# or a tf.Variable (and not a symbolic input tensor).
#
# The 'compute_input_gradient' method shows how to do this:
# pass inputs as a tf.Variable or a watched tensor,
# run the forward pass and loss computation under a GradientTape,
# and then compute gradients w.r.t the inputs.
#
# This code structure summarizes the essence of all the discussion and example code:
# - Model subclassing with MobileNet base
# - Proper gradient computation w.r.t inputs with tape.watch
# - Generating a suitable random input of shape (1, 180, 320, 3)
#
# This model and method will be compatible with TF 2.20.0 XLA compilation:
# Use:
# @tf.function(jit_compile=True)
# def compiled(x, y):
#     model = MyModel()
#     return model.compute_input_gradient(x, y, loss_fn)


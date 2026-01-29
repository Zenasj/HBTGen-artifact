# tf.random.uniform((2, 200, 200, 3), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(3, kernel_size=3, padding="same", name="conv_1")
        # For demonstration, no extra members needed here.
        
    def call(self, inputs, training=None):
        # On-call, produce conv output tensor
        return self.conv_1(inputs)

def my_model_function():
    # Create and compile model with run_eagerly=True to ensure eager execution,
    # so that tf.summary.image can consume eager tensors without error.
    model = MyModel()
    model.compile(
        optimizer=tf.optimizers.Adam(0.001), 
        loss=tf.losses.MeanAbsoluteError(),
        run_eagerly=True   # Important to produce EagerTensor output from call()
    )
    return model

def GetInput():
    # According to the example, typical input shape is (2, 200, 200, 3)
    # with float32 dtype, matching what conv2d expects.
    return tf.random.uniform((2, 200, 200, 3), dtype=tf.float32)

# ---
# ### Explanation and assumptions:
# - The original issue centers on `tf.summary.image` erroring when passed a symbolic Tensor (non-eager), as happens when using `tf.keras.Model.fit()` in graph mode. To fix this, the recommended approach is compiling the model with `run_eagerly=True`. This ensures that the output of `call()` is an eager tensor suitable for `tf.summary.image`.
# - The original code example showed a conv2d layer producing an output tensor of shape `[2, 200, 200, 3]` and issues logging its image summary.
# - We reproduce the core minimal model as `MyModel(tf.keras.Model)` with one Conv2D layer named `"conv_1"` configured with 3 output channels and kernel size 3, padding same, matching the original.
# - The `my_model_function()` returns the compiled model with `run_eagerly=True` explicitly to enable eager execution during training, matching the workaround suggested in the issue discussion.
# - The `GetInput()` helper returns a random float tensor with the required input shape `(2, 200, 200, 3)` resembling what was passed in the issue.
# - No callback is included here, as the main issue was about symbolic vs eager tensors in the context of `tf.summary.image`.
# - This model should also be fully compatible with TF 2.20.0 XLA compilation, as it uses only standard layers and `run_eagerly=True` (note that `run_eagerly=True` disables graph tracing so you might want to disable it for final serving).
# - This model structure and behavior capture the central problem and recommended solution described in the issue.
# If you want, I can also demonstrate a minimal example for how to properly log images from the model output using the `tf.summary.image` API with a custom training loop to avoid eager vs symbolic conflicts. Let me know!
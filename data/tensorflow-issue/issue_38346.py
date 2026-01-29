# tf.random.uniform((None, None, 12), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model wraps tf.signal.frame with fixed frame_length=2, frame_step=1, axis=1,
        # which was the core op causing loading issues in TF 2.2/2.3.
        # We emulate the original model: Input shape (None, None, 12).
        # To avoid errors on save/load, keep the model as a single Layer / Model
        # without empty reshape or ambiguous dtype issues.
        
        # No trainable weights needed as this is a signal preprocessing layer.
        pass

    def call(self, inputs):
        # inputs expected shape: (batch, time, features) = (None, None, 12)
        # We apply tf.signal.frame to frame on axis=1 (time dimension)
        # frame_length=2, frame_step=1 as in original reported code.
        framed = tf.signal.frame(inputs, frame_length=2, frame_step=1, axis=1)
        # output shape: (batch, frames, 2, features=12)
        # This matches the output shape seen in the issue model summary.
        return framed

def my_model_function():
    # Create and return the MyModel instance
    return MyModel()

def GetInput():
    # Return a random float32 tensor with shape (B, T, 12) for arbitrary batch and time.
    # We choose B=2, T=5 arbitrarily for testing.
    B, T, C = 2, 5, 12
    return tf.random.uniform((B, T, C), dtype=tf.float32)

# Notes / rationale:
# - The issue boils down to saving and loading Keras models that use tf.signal.frame,
#   which internally builds a reshape with an empty shape list (like []) causing dtype conflicts.
# - Earlier TF versions had trouble distinguishing the dtype of the shape tensor (float vs int)
#   leading to load errors.
# - Workarounds involved adding a Flatten or adding trainable params to force a non-empty shape,
#   or using tf.saved_model.save/load instead of model.save/load with h5.
# - The above class reproduces the core logic simply and should save/load cleanly on TF 2.20.0+.
# - For the other reshape-related issues described in the issue (reshape with [ ] shape tensor),
#   those are avoided here by not including the problematic reshape.
# - Input shape and dtype matches the user's original snippet.
# - The `GetInput` matches the input expected by the model.
# - This code is compatible with XLA compilation (`@tf.function(jit_compile=True)`),
#   as it uses supported ops and has no dynamic unsupported shapes.


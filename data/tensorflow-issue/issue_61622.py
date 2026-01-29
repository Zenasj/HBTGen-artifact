# tf.random.uniform((3, 300, 300, 192), dtype=tf.float32) ← Input shape inferred from provided example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The issue arises when a very large padding is used leading to overflow during multiplication:
        # Here we implement ZeroPadding2D with safe default padding,
        # but allow large padding input to illustrate the scenario.
        self.zero_padding = tf.keras.layers.ZeroPadding2D(
            padding=[[125091515651, False], [125091515651, 125091515651]]
            # This padding is taken directly from the example that caused overflow.
            # 'False' will be interpreted as 0 in the padding.
            # Note: This large padding will cause overflow if actually applied.
        )

    def call(self, inputs):
        # Defensive call to try applying zero padding
        # Because the input padding is huge, this can cause overflow in tf.pad op internally.
        # For demonstration, we catch this internally and return a message tensor instead of crashing.
        try:
            return self.zero_padding(inputs)
        except Exception as e:
            # Return a tensor with error string encoded as bytes to keep tf.function compatibility.
            # This is a workaround fallback so model still returns something
            err_msg = f"Padding error: {str(e)}"
            err_bytes = tf.strings.format("{}", err_msg)
            # Expand dims to produce a tensor output of shape (1,) with dtype string
            return err_bytes[None]
        

def my_model_function():
    # Return an instance of MyModel with the problematic padding setting encoded.
    return MyModel()

def GetInput():
    # Return a random tensor of float32 matching the input shape used in the example:
    # batch=3, height=300, width=300, channels=192
    return tf.random.uniform((3, 300, 300, 192), dtype=tf.float32)


# tf.random.uniform((B, None, 1), dtype=tf.float32) and a tf.constant of shape (B,) with dtype=tf.string as inputs

import tensorflow as tf

class InnerModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No additional layers, pass through lang input
        # This simulates the minimal processing example from issue
    def call(self, inputs, training=False):
        audio, lang = inputs
        # In the original example, the output is just 'lang'
        return lang

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.inner_model = InnerModel()

    def call(self, inputs, training=False):
        # inputs is a list or tuple of two tensors:
        # audio shape: (batch, None, 1), dtype=tf.float32
        # lang shape: (batch,) dtype=tf.string
        audio, lang = inputs

        # Pass inputs to inner model
        output = self.inner_model([audio, lang], training=training)
        return output

def my_model_function():
    # Return an instance of MyModel; matches the original model with InnerModel inside
    return MyModel()

def GetInput():
    # Create inputs matching shapes and dtypes expected:
    # audio: batch size 1, length 1000, 1 channel
    audio_input = tf.random.uniform((1, 1000, 1), dtype=tf.float32)
    # lang: shape (batch,), a string tensor with one language label
    lang_input = tf.constant(["fr"])  # batch size 1
    return [audio_input, lang_input]


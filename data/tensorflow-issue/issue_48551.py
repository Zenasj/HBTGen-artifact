# tf.constant(shape=(), dtype=tf.string) ← The input is a scalar string tensor.

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A model that lower-cases input strings using tf.strings.lower,
    demonstrating proper Unicode-aware lowercasing by specifying encoding='utf-8'.

    This addresses the bug where tf.strings.lower without encoding only lower-cases ASCII,
    and does not properly handle non-ASCII uppercase Unicode characters.
    """
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        # inputs: tf.Tensor of dtype tf.string, scalar or vector of strings
        # Use encoding='utf-8' to correctly handle non-ASCII characters
        lowered = tf.strings.lower(inputs, encoding='utf-8')
        return lowered

def my_model_function():
    # Instantiate and return the MyModel instance
    return MyModel()

def GetInput():
    # Return a tf.string tensor input compatible with the model, containing
    # Slovak text with accented uppercase characters to demonstrate correct lowering
    # Using a scalar string tensor as that matches example usage
    example_text = (
        "STARÝ KÔŇ NA HŔBE KNÍH ŽUJE TÍŠKO POVÄDNUTÉ RUŽE"  # Uppercase Slovak text with accents
    )
    return tf.constant(example_text, dtype=tf.string)


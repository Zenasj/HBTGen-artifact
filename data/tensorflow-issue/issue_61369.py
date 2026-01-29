# tf.random.uniform((B, 1), dtype=tf.int64) ← Input shape is (batch_size, 1), dtype int64, from the StringLookup input

import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class MyStringLookup(tf.keras.layers.StringLookup):
    def get_config(self):
        base_config = super().get_config()
        # Override get_config to include actual vocabulary to fix empty vocab on load issue
        custom = {"vocabulary": self.get_vocabulary()}
        return {**base_config, **custom}

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Use the fixed StringLookup with preserved vocabulary
        self.string_lookup = MyStringLookup(vocabulary=['a', 'b'])
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10)

    def call(self, inputs):
        x = self.string_lookup(inputs)
        x = self.flatten(x)
        return self.dense(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor of shape (batch_size, 1) with dtype int64 (as required by StringLookup input)
    # We'll sample random tokens from 'a', 'b', or OOV token (encoded as strings, but StringLookup expects string dtype input)
    # Note: From the original example, the model_input was dtype=int64 which actually conflicts with usual StringLookup inputs,
    #       as StringLookup expects string tensor inputs.
    #
    #       However, from the original code: model_input = tf.keras.Input(shape=(1,), dtype=tf.int64)
    #       And usage: StringLookup(vocabulary=['a','b'])(model_input)
    #
    #       This is unusual because StringLookup usually expects string inputs. Here, assume integer inputs (int64)
    #       that map to vocab strings ['a', 'b'], so we keep input dtype int64 and input shape (B,1).
    #
    # We'll create a random int64 tensor with values between 0 and 2 (to simulate token indices), shape (batch_size, 1).
    batch_size = 4
    input_tensor = tf.random.uniform(shape=(batch_size,1), minval=0, maxval=2, dtype=tf.int64)
    return input_tensor


# tf.random.uniform((B, 2), dtype=tf.float32), tf.constant with shape (B, 2), dtype=tf.string

import tensorflow as tf
from tensorflow import keras

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # StringLookup layer converts string input to integer indices.
        # This will embed strings "foo", "bar", "baz", "qux" to integer tokens 1..4.
        self.string_lookup = keras.layers.StringLookup(
            vocabulary=["foo", "bar", "baz", "qux"], mask_token=None, name='string_lookup'
        )
        # Concatenate float input and string embeddings (int indices)
        self.concatenate = keras.layers.Concatenate(name='concatenate')
        # Dense layers
        self.dense_1 = keras.layers.Dense(10, activation='relu', name='dense_1')
        self.output_layer = keras.layers.Dense(1, activation='sigmoid', name='output')

    def call(self, inputs, training=False):
        input_float, input_string = inputs
        # string_lookup outputs int tensors, compatible for Concatenate with float tensor
        string_embedding = self.string_lookup(input_string)
        # Concatenate float input (float32) and string embedding (int32)
        # Cast string_embedding to float32 for concatenation since dense layers expect float input
        string_embedding = tf.cast(string_embedding, dtype=tf.float32)
        concatenated = self.concatenate([input_float, string_embedding])
        x = self.dense_1(concatenated)
        output = self.output_layer(x)
        return output

def my_model_function():
    # Return an instance of MyModel with initialized layers.
    model = MyModel()
    # Build the model by calling it once with dummy input to create weights
    # This avoids potential issues in TF functional contexts
    dummy_float = tf.zeros((1, 2), dtype=tf.float32)
    dummy_string = tf.constant([["foo", "bar"]], dtype=tf.string)
    _ = model((dummy_float, dummy_string))
    # Compile the model with the same settings as original example
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def GetInput():
    # Generate a batch of 2 samples
    float_data = tf.random.uniform((2, 2), dtype=tf.float32)
    string_data = tf.constant([["foo", "bar"], ["baz", "qux"]], dtype=tf.string)
    return (float_data, string_data)


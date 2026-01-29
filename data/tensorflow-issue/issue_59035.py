# tf.random.uniform((B, 1), dtype=tf.string)  ← Assumption: Input is a batch of string tensors with shape (batch_size, 1)

import tensorflow as tf

@tf.function
def mapp(string):
    # Example mapping: convert string to lower case
    return tf.strings.lower(string)

class MyModel(tf.keras.Model):
    def __init__(self, separator=" "):
        super(MyModel, self).__init__()
        self.separator = separator

    def call(self, inputs):
        """
        inputs: tf.Tensor of shape (batch_size, 1) and dtype tf.string

        This layer aims to process strings inside the TensorFlow model by:
        1. Splitting strings by separator into RaggedTensor
        2. Applying a mapped function (e.g. lowercase) to each token using tf.map_fn
        3. Joining back the tokens into a single string per example

        Due to TensorFlow limitations, tf.map_fn is currently used to simulate
        iteration over RaggedTensors tokens.
        """
        # Split input string tensor by separator → RaggedTensor (batch_size, None)
        splits = tf.strings.split(inputs, sep=self.separator)  

        # tf.map_fn applies mapp to each token in splits (ragged)
        # splits is RaggedTensor, but map_fn expects a Tensor, so map_fn works on splits.flat_values
        mapped_flat_values = tf.map_fn(mapp, splits.flat_values)

        # Construct RaggedTensor again with mapped values
        mapped = tf.RaggedTensor.from_row_splits(mapped_flat_values, splits.row_splits)

        # Join mapped tokens back with separator, result shape (batch_size,)
        joined = tf.strings.reduce_join(mapped, axis=1, separator=self.separator)

        # Expand dims to keep shape consistent with input (batch_size, 1)
        output = tf.expand_dims(joined, axis=-1)

        return output

def my_model_function():
    # Return an instance of MyModel with default separator " "
    return MyModel()

def GetInput():
    # Generate a batch of 4 example string tensors with shape (4, 1)
    # Example sentences to simulate tokens joined by spaces
    sentences = [
        "Hello TensorFlow",
        "This is a Test",
        "StRing PROCESSing",
        "GPU utilization"
    ]
    # Convert to tf.Tensor of shape (batch_size=4, 1) dtype=tf.string
    input_tensor = tf.constant(sentences, shape=(4, 1), dtype=tf.string)
    return input_tensor


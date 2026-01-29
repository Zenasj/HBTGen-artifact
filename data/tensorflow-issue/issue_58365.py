# tf.random.uniform((B,), dtype=tf.string) ← input batch of string tensors, shape unknown (batch size only)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # StringLookup layer with no pre-defined vocabulary, adapting expects string inputs.
        self.string_lookup = tf.keras.layers.StringLookup()
        # Embedding layer sized dynamically after adapt
        self.embedding = None

    def adapt(self, dataset_strings):
        # Adapt the StringLookup on the dataset strings (dataset of (batch_size,) string tensors)
        self.string_lookup.adapt(dataset_strings)
        # After adapt, create embedding layer of vocabulary size from string_lookup
        vocab_size = self.string_lookup.vocabulary_size()
        self.embedding = tf.keras.layers.Embedding(vocab_size, 4)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # inputs: tf.Tensor of dtype = string and shape (batch_size,)
        # Use StringLookup to convert strings to integer indices
        x = self.string_lookup(inputs)
        # Pass integer indices to embedding layer
        return self.embedding(x)

def my_model_function():
    model = MyModel()
    # For demonstration/adaptation, create a dummy dataset of strings to adapt
    example_strings = tf.data.Dataset.from_tensor_slices(["a", "b", "c", "d"]).batch(2)
    model.adapt(example_strings)
    return model

def GetInput():
    # Generate a batch of strings matching the model's expected input
    # Here, batch size 2 with example strings matching the vocab used for adapt
    batch_strings = tf.constant(["a", "c"])
    return batch_strings


# tf.random.uniform((None, 16), dtype=tf.int32) ← Input shape and dtype inferred from input_signature in code snippet

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, base_model=None, vocab_keys=None, vocab_values=None, default_value="UNKNOWN"):
        super(MyModel, self).__init__()
        # The original issue describes a model whose output is passed through a StaticHashTable lookup
        # in a Lambda layer to convert model_id to labels.
        # Since the full base model is not provided, we will allow passing a base_model.
        # If None, use a dummy simple model producing int32 outputs of shape (None, 16).
        if base_model is None:
            self.base_model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(16,), dtype=tf.int32),
                # Just an example dense layer outputting int32 IDs to lookup
                tf.keras.layers.Lambda(lambda x: tf.cast(tf.math.mod(x, 10), tf.int32))
            ])
        else:
            self.base_model = base_model

        # Build a StaticHashTable for mapping IDs to labels
        # If vocab keys and values are not provided, create a dummy example mapping
        if vocab_keys is None or vocab_values is None:
            # Example default static hash table map
            # mapping IDs 0-9 to string labels
            keys = tf.constant([i for i in range(10)], dtype=tf.int64)
            values = tf.constant([f"label_{i}" for i in range(10)])
            vocab_keys = keys
            vocab_values = values

        initializer = tf.lookup.KeyValueTensorInitializer(
            keys=vocab_keys,
            values=vocab_values,
            key_dtype=tf.int64,
            value_dtype=tf.string
        )
        self.lookup_table = tf.lookup.StaticHashTable(initializer, default_value=default_value)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 16), dtype=tf.int32)])
    def call(self, inputs):
        # Forward through base model
        model_output = self.base_model(inputs)  # Expected int32 tensor
        # Lookup string labels via StaticHashTable - need to cast keys to int64 as lookup requires that
        labels = self.lookup_table.lookup(tf.cast(model_output, tf.int64))
        return labels

def my_model_function():
    # Return an instance of MyModel with default dummy mapping and base model
    return MyModel()

def GetInput():
    # Return random int32 input tensor of shape (batch_size=4, 16), values in [0, 9]
    return tf.random.uniform((4, 16), minval=0, maxval=10, dtype=tf.int32)


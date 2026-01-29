# tf.RaggedTensor with shape=(B, None) for input sequences of variable length

import tensorflow as tf

class CustomMean(tf.keras.layers.Layer):
    def __init__(self, axis=None):
        super(CustomMean, self).__init__()
        # Support ragged inputs explicitly (needed for some ops)
        self._supports_ragged_inputs = True
        self.axis = axis

    def call(self, inputs, **kwargs):
        return tf.reduce_mean(inputs, axis=self.axis)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding layer supporting ragged inputs natively in TF 2.x
        self.embedding = tf.keras.layers.Embedding(input_dim=100, output_dim=4)
        # Using custom mean layer to pool embeddings along ragged axis
        self.pooled = CustomMean(axis=1)
        self.classifier = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, inputs, training=False):
        """
        inputs: tf.RaggedTensor of dtype tf.int32 or tf.int64 with shape (batch_size, None)
        """
        x = self.embedding(inputs)             # Output shape: (B, None, 4), ragged preserved
        x = self.pooled(x)                     # Shape: (B, 4), pooling over variable length dimension
        return self.classifier(x)              # Shape: (B, 2), classification logits

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a RaggedTensor input compatible with MyModel
    # Assume batch size B=3, variable sequence lengths
    values = tf.constant([1, 2, 3, 1, 2, 1, 2, 3, 4], dtype=tf.int32)
    row_splits = tf.constant([0, 3, 5, 9], dtype=tf.int64)  # splits for 3 sequences: lengths 3,2,4
    ragged_input = tf.RaggedTensor.from_row_splits(values, row_splits)
    return ragged_input


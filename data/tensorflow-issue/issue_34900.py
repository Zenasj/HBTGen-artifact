# tf.random.uniform((B, 512), dtype=tf.int64) ← input is int64 tensor with shape (batch_size, 512)

import tensorflow as tf
from tensorflow.keras import layers, Model, Input

# Placeholder for TFBertModel from transformers library
# In practice, replace this with the actual import:
# from transformers import TFBertModel

class DummyTFBertModel(tf.keras.layers.Layer):
    """
    This dummy layer simulates the output of a TFBertModel.
    The real TFBertModel returns a tuple with first element
    of shape (batch_size, seq_len, hidden_size).
    Here seq_len=512 and hidden_size=768 to match the original model.
    """
    def __init__(self):
        super().__init__()
        self.seq_len = 512
        self.hidden_size = 768

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # Simulate output tensor: shape (batch_size, seq_len, hidden_size)
        dummy_output = tf.zeros((batch_size, self.seq_len, self.hidden_size), dtype=tf.float32)
        # The real TFBertModel outputs a tuple with at least the last hidden states as first element
        return (dummy_output,)


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Input specification: shape=(512,), dtype=int64 required as in original
        # Initialize a dummy/simulated BERT model to replicate structure
        self.bert = DummyTFBertModel()
        self.dropout = layers.Dropout(0.1)
        self.flatten = layers.Flatten()
        self.classifier = layers.Dense(units=5, activation="softmax")

    def call(self, inputs, training=False):
        # inputs: int64 tensor shape (batch_size, 512)
        bert_outputs = self.bert(inputs)
        # bert_outputs[0] expected shape: (batch_size, 512, 768)
        x = bert_outputs[0]
        x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    model = MyModel()

    # Compile the model similar to original snippet
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-6, epsilon=1e-08, clipnorm=1.0)
    # Note: original loss uses from_logits=True but final layer is softmax,
    # which is typically incompatible; replicating original however.
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    return model


def GetInput():
    # Return a random input tensor with shape (batch_size, 512) and dtype int64
    # batch_size = 2 for example purposes
    batch_size = 2
    return tf.random.uniform(shape=(batch_size, 512), minval=0, maxval=1000, dtype=tf.int64)


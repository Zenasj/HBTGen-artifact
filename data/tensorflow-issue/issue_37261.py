# tf.random.uniform((B,), dtype=tf.string) ← Input is a 1D batch of strings (texts) for text classification

import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using the same pretrained text embedding as in the issue
        self.embedding = hub.KerasLayer(
            "https://tfhub.dev/google/nnlm-en-dim128/2",
            trainable=False,
            dtype=tf.string
        )
        # Simple classifier layers similar to estimator's hidden_units=[64,32]
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        # Output layer for binary classification (2 classes)
        self.out = tf.keras.layers.Dense(2)  # logits (no activation)

    def call(self, inputs, training=False):
        """
        inputs: tf.Tensor of shape (batch_size,) and dtype string
        Returns: logits of shape (batch_size, 2)
        """
        x = self.embedding(inputs)  # (batch_size, 128)
        x = self.dense1(x)
        x = self.dense2(x)
        logits = self.out(x)
        return logits

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Since this is similar to the original code using estimator with Adagrad lr=0.003,
    # we mimic optimizer setup here if training were to be done outside:
    model.compile(
        optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.003),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a batch of string inputs like the "html" text column from train/test
    # Let's return a batch of 4 example texts (random placeholders)
    texts = [
        "this is an example of an e-commerce website text description",
        "news article about current events and politics",
        "product page with description and pricing info",
        "a blog post about cooking recipes and tips"
    ]
    return tf.constant(texts, dtype=tf.string)


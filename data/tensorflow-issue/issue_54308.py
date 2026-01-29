# tf.random.uniform((B,), dtype=tf.string) ← Input is a batch of strings (tweets)

import tensorflow as tf
import tensorflow_hub as hub

class MyModel(tf.keras.Model):
    def __init__(self, num_labels):
        super().__init__()
        # Load BERT preprocessing and encoder layers from TF Hub
        self.bert_preprocess = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            name="bert_preprocess"
        )
        self.bert_encoder = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
            trainable=True,  # Typically BERT encoder is fine-tuned
            name="bert_encoder"
        )
        # Dropout and classification head layers
        self.dropout = tf.keras.layers.Dropout(0.1, name="dropout")
        self.classifier = tf.keras.layers.Dense(num_labels, activation="softmax", name="output")

    def call(self, inputs, training=False):
        # inputs: batch of strings of shape (batch_size,)
        x = self.bert_preprocess(inputs)  # Preprocess input string tensor for BERT
        bert_outputs = self.bert_encoder(x)
        pooled_output = bert_outputs["pooled_output"]  # [batch_size, 768]
        x = self.dropout(pooled_output, training=training)
        logits = self.classifier(x)
        return logits


def my_model_function():
    # Assumption: number of labels is 2 for binary classification by default.
    # In practice, this should match the dataset.
    num_labels = 2
    return MyModel(num_labels)


def GetInput():
    # Return a batch of string tensors as input, shape (batch_size,).
    # We'll generate 4 dummy samples of arbitrary strings.
    batch_size = 4
    sample_texts = [
        "This is a sample tweet for testing.",
        "TensorFlow models are great for NLP tasks.",
        "Apple M1 GPU with tensorflow-metal is fast!",
        "How to fix Check failed: IsAligned() ptr error?"
    ]
    # Pad or truncate to batch_size
    sample_texts = sample_texts[:batch_size]
    # Convert list of strings to Tensor of dtype tf.string
    return tf.constant(sample_texts, dtype=tf.string)


# tf.random.uniform((B, ) with string inputs (variable-length scalar strings)

import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Create a TextVectorization layer with a custom standardization function as in example
        self.vectorize_layer = tf.keras.layers.TextVectorization(
            standardize=self.tf_lower_and_split_punct
        )
        # Adapt on some sample text to initialize the vocabulary
        sample_text = tf.constant(
            [
                "Był to świetny pomysł, bo punktował Prawo i Sprawiedliwość tam, gdzie jest ono najsłabsze, "
                "mimo że udaje najsilniejsze. Uderzał w wizerunek państwa dobrobytu, które nikogo nie zostawia "
                "z tyłu i wyrównuje szanse. Tutaj mamy pewnego rodzaju déjà vu."
            ]
        )
        self.vectorize_layer.adapt(sample_text)

    def tf_lower_and_split_punct(self, text):
        # Replicated from the example: strip whitespace, join with [START] and [END] tokens
        text = tf.strings.strip(text)
        text = tf.strings.join(["[START]", text, "[END]"], separator=" ")
        return text

    def get_vocabulary(self):
        # Custom get_vocabulary method that tries to decode bytes,
        # falling back to ISO-8859-1 on errors (UnicodeDecodeError was an issue on Windows)
        keys, values = self.vectorize_layer._lookup_layer.lookup_table.export()
        vocab = []
        for i in keys:
            try:
                vocab.append(i.numpy().decode("utf-8"))
            except Exception:
                vocab.append(i.numpy().decode("ISO-8859-1"))
        return vocab

    def call(self, inputs):
        # Forward pass applies the text vectorization layer
        return self.vectorize_layer(inputs)


def my_model_function():
    # Return an instance of MyModel including initialization and adapted vocabulary
    return MyModel()


def GetInput():
    # Returns a batch of strings as input, matching the expected input of MyModel
    sample_sentences = [
        "To jest przykładowe zdanie.",
        "Inny przykład tekstu do wektoryzacji.",
        "Zażółć gęślą jaźń."
    ]
    # Tensor of shape (batch_size,) with dtype string
    return tf.constant(sample_sentences)


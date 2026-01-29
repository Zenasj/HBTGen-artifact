# tf.random.uniform((B,)) ← Input shape is a batch of strings (text samples) as 1D tensors of dtype tf.string

import tensorflow as tf
from tensorflow.keras import layers

# Custom standardization function used for TextVectorization
def custom_standardization(input_data):
    import re
    import string
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')

class MyModel(tf.keras.Model):
    def __init__(self, max_features=10000, embedding_dim=16, sequence_length=250):
        super().__init__()
        # Text vectorization layer - converts raw text string to int sequences
        self.vectorize_layer = layers.TextVectorization(
            standardize=custom_standardization,
            max_tokens=max_features,
            output_mode='int',
            output_sequence_length=sequence_length)

        # Embedding and classifier layers following the original sequential model structure
        self.embedding = layers.Embedding(max_features + 1, embedding_dim)
        self.dropout1 = layers.Dropout(0.2)
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.dropout2 = layers.Dropout(0.2)
        self.dense = layers.Dense(1)  # Output logits for binary classification

        # Internal flag to check if vectorize_layer is adapted, since adapt must be called before use
        self._adapted = False

    def adapt(self, text_dataset):
        # Adapt the vectorization layer on a dataset consisting of text strings
        self.vectorize_layer.adapt(text_dataset)
        self._adapted = True

    def call(self, inputs, training=False):
        # inputs expected as batch of raw text strings: shape (batch_size,)
        if not self._adapted:
            raise RuntimeError("Vectorization layer is not adapted. Call adapt() before using the model.")

        # Vectorize text input to integer sequence tokens
        x = self.vectorize_layer(inputs)
        x = self.embedding(x)
        x = self.dropout1(x, training=training)
        x = self.global_avg_pool(x)
        x = self.dropout2(x, training=training)
        x = self.dense(x)
        return x


def my_model_function():
    # Instantiate MyModel with default parameters
    model = MyModel()

    # For demonstration, we mimic adapting the vectorize_layer on some dummy data
    # In practice, call `model.adapt(train_text)` with real training text dataset (tf.data.Dataset)
    sample_text = tf.data.Dataset.from_tensor_slices([
        "This is a sample sentence.",
        "TensorFlow is great for ML.",
        "This text is for vectorization adapt step."
    ])
    model.adapt(sample_text)

    # Compile the model similarly to the original example
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0)]
    )
    return model

def GetInput():
    # Return a batch of raw text samples as input tensor
    # For example: batch size 4 of strings
    sample_texts = [
        "This movie was fantastic! I loved it.",
        "Terrible movie. Waste of time.",
        "I think it was okay, could be better.",
        "An excellent film with great visuals!"
    ]
    return tf.constant(sample_texts, dtype=tf.string)


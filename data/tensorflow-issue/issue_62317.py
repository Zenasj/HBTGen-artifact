# tf.random.uniform(()) ← Text input is a scalar string tensor, no fixed numerical shape

import tensorflow as tf
from tensorflow.keras.layers import Input, TextVectorization, Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # TextVectorization adapted on sample data from the issue:
        self.vectorizer = TextVectorization(
            max_tokens=20,
            output_mode='int',
            output_sequence_length=5,
            pad_to_max_tokens=True
        )
        # We will adapt the vectorizer in __init__ with example data:
        example_text = tf.constant([
            "This is shit",
            "I hate this",
            "Great I love this!!!"
        ])
        self.vectorizer.adapt(example_text)

        # Embedding layer, updated to use current Keras API without `input_length` (deprecated)
        self.embedding = Embedding(input_dim=20, output_dim=100)

        self.pooling = GlobalAveragePooling1D()
        self.output_layer = Dense(1, activation='sigmoid', name='output')

    def call(self, inputs):
        # inputs is a batch of strings: shape (batch_size, 1) or (batch_size,)
        x = self.vectorizer(inputs)
        x = self.embedding(x)
        x = self.pooling(x)
        output = self.output_layer(x)
        return output


def my_model_function():
    # Returns an instance of the model compiled with RMSprop optimizer
    model = MyModel()

    # Based on the issue discussion:
    # Use optimizer config with updated parameter names compatible with Keras 3+ / TF 2.20
    optimizer_config = {'class_name': 'RMSprop', 'config': {'learning_rate': 0.0001}}
    optimizer = tf.keras.optimizers.get(optimizer_config)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def GetInput():
    # Return a batch of random string tensors matching input shape expected by MyModel
    # Input shape expected: a batch of shape (batch_size,), dtype string.
    # We use some example sentences similar to those used for adapting the vectorizer
    texts = [
        "I love TensorFlow",
        "This model is great",
        "Keras makes life easier",
        "Bug fixes are important",
    ]
    # Convert to a tensor of shape (batch_size,)
    return tf.constant(texts)


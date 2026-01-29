# tf.random.uniform((B,), dtype=tf.string) ← The input is a batch of text strings for sentiment analysis

import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorflow_hub as hub

class MyModel(tf.keras.Model):
    def __init__(self, embedding="https://tfhub.dev/google/nnlm-en-dim128/1", name="MyModel", **kwargs):
        super(MyModel, self).__init__(name=name, **kwargs)
        # Embedding layer from TF Hub for text input, output dimension is 128
        self.embedding_layer = hub.KerasLayer(embedding, trainable=True, dtype=tf.string)
        # Two dense layers: first with ReLU, second with sigmoid for binary classification
        self.dense1 = Dense(16, activation='relu')
        self.dense2 = Dense(1, activation='sigmoid')
        
    @tf.function
    def call(self, inputs):
        # Forward pass: embed text inputs -> dense relu -> dense sigmoid
        x = self.embedding_layer(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of the model with default parameters
    return MyModel()

def GetInput():
    # Create a sample batch of sentences as tf.string tensor input
    sample_sentences = [
        "I loved this movie, it was fantastic!",
        "The film was terrible and boring.",
        "Absolutely wonderful experience.",
        "Not good, I would not recommend.",
        "An okay movie, nothing special."
    ]
    # Convert to tf.Tensor of dtype string
    return tf.constant(sample_sentences)


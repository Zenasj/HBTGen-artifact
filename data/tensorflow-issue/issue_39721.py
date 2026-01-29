# tf.random.uniform((B, 192), dtype=tf.int32) ← Input shape: batch size B, sequence length 192, int32 tokens

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from transformers import TFDistilBertModel, DistilBertTokenizer
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self, transformer_name='distilbert-base-multilingual-cased', max_len=192):
        super().__init__()
        self.max_len = max_len
        # Load pretrained DistilBERT transformer model
        self.transformer = TFDistilBertModel.from_pretrained(transformer_name)
        # Classification head
        self.classifier = Dense(1, activation='sigmoid')
        # Compile model with Adam optimizer and binary crossentropy loss
        self.compile(optimizer=Adam(learning_rate=1e-5),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

    def call(self, inputs, training=False):
        """
        Forward pass:
        inputs: [batch_size, max_len] int32 tensor of token ids
        """
        # Extract sequence output from transformer (batch_size, seq_len, hidden_dim)
        sequence_output = self.transformer(inputs)[0]
        # Take CLS token embedding (first token embedding for classification)
        cls_token = sequence_output[:, 0, :]  # shape: (batch_size, hidden_dim)
        # Pass through classification head
        out = self.classifier(cls_token)  # shape: (batch_size, 1)
        return out

def my_model_function():
    """
    Instantiates and returns the MyModel with the pretrained distilbert-base-multilingual-cased transformer,
    max sequence length 192, compiled for binary classification.
    """
    return MyModel(transformer_name='distilbert-base-multilingual-cased', max_len=192)

def GetInput():
    """
    Returns a random tensor input compatible with MyModel.
    This is an integer tensor shaped [batch_size, max_len] with token ids expected by DistilBERT tokenizer.
    Here, batch_size is chosen as 16 (typical batch size).
    """
    batch_size = 16
    max_len = 192
    # DistilBERT's vocab size is approx 120000 tokens, using 100,000 for safety.
    vocab_size = 100000
    # Random integers in [0, vocab_size) representing token ids
    random_input = tf.random.uniform((batch_size, max_len), minval=0, maxval=vocab_size, dtype=tf.int32)
    return random_input


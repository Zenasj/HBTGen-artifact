# tf.random.uniform((B,), dtype=tf.string) ← Input is a batch of variable-length strings (text input)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub

# Because original code references the ALBERT preprocessor TF Hub model and custom transformer block,
# This reconstructed model matches the structure given in the issue and uses a hub.KerasLayer for preprocessing.

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        # x shape: (batch_size, sequence_length)
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class MyModel(tf.keras.Model):
    def __init__(self, num_classes=4):
        super(MyModel, self).__init__()
        # Using the same preprocessing layer as original: ALBERT English preprocess v3
        # Assumption: The local path './albert_en_preprocess_3' is replaced with standard TFHub URL for universal usage,
        # since no local files can be referenced here.
        self.preprocessor_layer = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/albert_en_preprocess/3", trainable=False
        )
        # To get vocab_size, attempt to load the tokenizer from hub (cannot run here, so hardcode approximation)
        # Original code attempts: preprocessor.tokenize.get_special_tokens_dict()['vocab_size']
        # ALBERT vocab size is about 30000 tokens, so use 30000 as a safe assumption.
        vocab_size = 30000
        embed_dim = 32
        num_heads = 2
        ff_dim = 32
        max_len = 128  # Assumed max length for input tokens

        # Embedding for tokens and positions (sequence length max_len)
        self.embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.dropout1 = layers.Dropout(0.1)
        self.dense1 = layers.Dense(20, activation="relu")
        self.dropout2 = layers.Dropout(0.1)
        self.classifier = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        """
        inputs: tf.Tensor of dtype tf.string and shape (batch_size,)
        """
        # Preprocess the raw string text input to get token IDs (int32 tensors)
        preprocessed = self.preprocessor_layer(inputs)
        encoder_inputs = preprocessed["input_word_ids"]  # shape (batch_size, sequence_length)

        # For safe fixed length max_len, pad or truncate:
        # Note: The preprocessor outputs fixed length, so we trust shape here
        x = self.embedding_layer(encoder_inputs)
        x = self.transformer_block(x, training=training)
        x = self.global_avg_pool(x)
        x = self.dropout1(x, training=training)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        output = self.classifier(x)
        return output


def my_model_function():
    # Return an instance of MyModel with default parameters for 4 classes as in original issue
    model = MyModel(num_classes=4)
    # Compile model similarly to original
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    return model


def GetInput():
    # Return a batch of random string tensor inputs that simulate text strings
    # Since model input dtype is tf.string and shape (batch_size,), generate batch_size=2 arbitrary strings
    # These strings can be any English sentences or gibberish tokens
    inputs = tf.constant(
        [
            "This is a sample input sentence.",
            "Another example sentence for testing."
        ],
        dtype=tf.string,
    )
    return inputs


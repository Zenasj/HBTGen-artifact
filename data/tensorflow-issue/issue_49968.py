# tf.random.uniform((B,), dtype=tf.string) ← The model expects a batch of variable-length string inputs (shape=())

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Placeholder implementations of TransformerBlock and TokenAndPositionEmbedding inferred from typical transformer structure,
# since original definitions were not included in the issue.

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
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

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=5001):
        super().__init__()
        import tensorflow_hub as hub

        # Load the preprocessor layer from TF Hub.
        # In the original code, this was "./albert_en_preprocess_3"
        # We infer this is a KerasLayer that takes string input and returns dict with 'input_word_ids'.
        self.preprocessor_layer = hub.KerasLayer("./albert_en_preprocess_3")

        # Get vocab size from the preprocessor tokenization info.
        # This is somewhat inferred, since in code it was:
        # vocab_size = preprocessor.tokenize.get_special_tokens_dict()['vocab_size'].numpy()
        # Here we emulate it with a placeholder vocab size to keep model runnable.
        try:
            preprocessor = hub.load("./albert_en_preprocess_3")
            self.vocab_size = int(preprocessor.tokenize.get_special_tokens_dict()['vocab_size'].numpy())
        except Exception:
            # Fallback vocab size if preprocessor loading is not available at runtime.
            self.vocab_size = 30000

        # Model hyperparameters matching code snippet
        self.embed_dim = 32
        self.num_heads = 2
        self.ff_dim = 32

        # Since input_word_ids shape is dynamic, assume max sequence length is fixed for embedding layers
        # In the original code, it used encoder_inputs.shape[1], but shape[1] may be None at init,
        # so we use 128 as a reasonable fixed max sequence length.
        max_seq_len = 128

        self.embedding_layer = TokenAndPositionEmbedding(max_seq_len, self.vocab_size, self.embed_dim)
        self.transformer_block = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dense_relu = layers.Dense(512, activation="relu")
        self.dense_softmax = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        """
        inputs: tf.Tensor of shape (batch_size,) dtype string
        """
        # Pass string inputs through preprocessor_layer (TF Hub/KerasLayer)
        # which outputs a dictionary with key 'input_word_ids'.
        processed = self.preprocessor_layer(inputs)
        encoder_inputs = processed['input_word_ids']  # shape (batch_size, seq_len), int32

        x = self.embedding_layer(encoder_inputs)
        x = self.transformer_block(x, training=training)
        x = self.global_pool(x)
        x = self.dense_relu(x)
        return self.dense_softmax(x)


def my_model_function():
    # Return an instance of MyModel with default number of classes (5001).
    # Note: The original code compiled model; for TF2 saved models and TFLite, compilation may not be necessary here.
    model = MyModel(num_classes=5001)
    # Compile is often required for training but not strictly for inference or TFLite conversion.
    # Include compilation to match original snippet:
    from tensorflow.keras.optimizers import Adam
    from tensorflow.python.keras.metrics import sparse_top_k_categorical_accuracy

    def acc_top4(y_true, y_pred):
        return sparse_top_k_categorical_accuracy(y_true, y_pred, k=4)

    def acc_top8(y_true, y_pred):
        return sparse_top_k_categorical_accuracy(y_true, y_pred, k=8)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["acc", acc_top4, acc_top8],
    )
    return model


def GetInput():
    # Return a batch of random string tensor input to MyModel.
    # Since the model expects batch dimension with dtype string scalars per element,
    # generate a batch of 4 dummy strings.

    # This matches input shape: (batch_size,), dtype=tf.string
    batch_size = 4
    dummy_strings = tf.constant([
        "example sentence one",
        "another example sentence",
        "tensorflow lite input test",
        "final input string",
    ], dtype=tf.string)
    # In case more or fewer needed, resize accordingly.

    return dummy_strings


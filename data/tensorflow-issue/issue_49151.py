# tf.random.uniform((B,)) ← Input is a batch of variable-length string tensors, shape (batch_size,) of dtype tf.string
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub
from tensorflow import keras

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        # x shape: (batch_size, seq_len, embed_dim)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len, projection_dim)

    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, embed_dim)
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # back to (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))  # concat all heads
        output = self.combine_heads(concat_attention)
        return output

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        # Use builtin MultiHeadAttention in TF 2.4 or 2.5; else fallback custom implementation.
        if tf.__version__.startswith('2.4') or tf.__version__.startswith('2.5'):
            self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        else:
            self.att = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        if tf.__version__.startswith('2.4') or tf.__version__.startswith('2.5'):
            attn_output = self.att(inputs, inputs)
        else:
            attn_output = self.att(inputs)
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
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class MyModel(tf.keras.Model):
    # This model encapsulates the transformer defined in the issue,
    # including preprocessing with ALBERT preprocessing layer
    # Takes batch of string inputs and outputs softmax predictions over num_classes.

    def __init__(self, num_classes=4):
        super(MyModel, self).__init__()
        self.embed_dim = 32
        self.num_heads = 2
        self.ff_dim = 32

        # Load TF Hub ALBERT English preprocessor
        self.preprocessor_file = "./albert_en_preprocess_3"  # Should be replaced with a real path or TFHub URL if used outside
        self.preprocessor_layer = hub.KerasLayer(self.preprocessor_file)
        # We rely on the loaded hub module to get vocabulary size dynamically
        preprocessor = hub.load(self.preprocessor_file)
        try:
            # This works if the hub module exposes the tokenizer in this way
            self.vocab_size = int(preprocessor.tokenize.get_special_tokens_dict()['vocab_size'].numpy())
        except Exception:
            # Fallback: heuristic vocab size to avoid crash - user should update as needed
            self.vocab_size = 30000

        # For simplicity, assume maxlen = fixed length from preprocessor_layer output shape
        # Since preprocessor_layer output shape is dynamic tensor, use 128 as a safe default maxlen
        self.maxlen = 128

        self.embedding_layer = TokenAndPositionEmbedding(self.maxlen, self.vocab_size, self.embed_dim)
        self.transformer_block = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dense_relu = layers.Dense(32, activation="relu")
        self.classifier = layers.Dense(num_classes, activation="softmax")

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def call(self, inputs, training=False):
        # inputs: batch of strings (shape [batch_size])
        encoder_inputs = self.preprocessor_layer(inputs)['input_word_ids']  # shape (batch_size, seq_len)
        # When seq_len varies, padding may be applied by preprocessor
        # We clip or pad to self.maxlen for embedding + transformer
        seq_len = tf.shape(encoder_inputs)[1]
        # Pad or slice encoder_inputs to maxlen
        if seq_len < self.maxlen:
            padding = self.maxlen - seq_len
            encoder_inputs = tf.pad(encoder_inputs, [[0, 0], [0, padding]])
        else:
            encoder_inputs = encoder_inputs[:, :self.maxlen]

        x = self.embedding_layer(encoder_inputs)  # (batch_size, maxlen, embed_dim)
        x = self.transformer_block(x, training=training)
        x = self.global_pool(x)
        x = self.dense_relu(x)
        outputs = self.classifier(x)
        return outputs

def my_model_function():
    # Returns an instance of the transformer model with 4 classes (default from example)
    return MyModel(num_classes=4)

def GetInput():
    # Return a random batch of string tensor inputs compatible with the ALBERT preprocessing layer
    # Since the model expects tf.string tensors of shape [batch_size] (batch of sentences),
    # here we mock a batch of size 2 with dummy sentences for stable input shape.
    dummy_sentences = tf.constant([
        "This is a sample input sentence.",
        "Here is another example input to the model."
    ])
    return dummy_sentences


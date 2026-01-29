# tf.random.uniform((B,), dtype=tf.string)  ← Input is a batch of strings (shape=(batch,), dtype=string)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Note: The original issue uses a tf.hub preprocessing layer that converts string input
# to token IDs internally. Since tf.hub layers aren't included here, 
# we simulate a minimal preprocessor with lookup table and simple tokenization.
# In practice, you'd replace this with the real preprocessor as a tf.keras Layer.

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

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
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
    def __init__(self, num_classes=5001, max_seq_len=128, vocab_size=10000):
        """
        Args:
            num_classes: number of output classes for classification
            max_seq_len: max length of token ids sequence after preprocessing
            vocab_size: vocabulary size of tokens after lookup
        """
        super(MyModel, self).__init__()
        
        # Simulate token lookup by a simple lookup table
        keys = tf.constant([f"word{i}" for i in range(vocab_size-5)])  # leave 5 OOV tokens
        values = tf.constant(list(range(vocab_size-5)), dtype=tf.int64)
        init = tf.lookup.KeyValueTensorInitializer(keys, values)
        self.table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets=5)
        
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        
        self.embedding_dim = 32
        self.num_heads = 2
        self.ff_dim = 32
        
        # The input here is string tensor (batch,)
        
        # Preprocessor layer equivalent:
        # 1) tokenize strings by splitting on space (simulate)
        # 2) lookup tokens to ids using vocab table
        # 3) pad/truncate to max_seq_len
        
        self.embedding_layer = TokenAndPositionEmbedding(max_seq_len, vocab_size, self.embedding_dim)
        self.transformer_block = TransformerBlock(self.embedding_dim, self.num_heads, self.ff_dim)
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(512, activation="relu")
        self.classifier = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        # inputs: batch of strings with shape (batch,)
        
        # 1) Tokenize by splitting on spaces
        tokens = tf.strings.split(inputs, sep=' ')  # RaggedTensor shape = (batch, None)
        
        # 2) Lookup tokens to token ids
        # Since StaticVocabularyTable requires rank-1 and returns rank-1,
        # we flatten tokens to 1D, lookup, then reshape back.
        flat_tokens = tokens.flat_values  # flatten token strings
        token_ids_flat = self.table.lookup(flat_tokens)
        token_ids = tf.RaggedTensor.from_row_splits(token_ids_flat, tokens.row_splits)
        
        # 3) Pad or truncate sequences to max_seq_len
        token_ids_padded = token_ids.to_tensor(default_value=0, shape=[None, self.max_seq_len])
        token_ids_padded = token_ids_padded[:, :self.max_seq_len]

        # 4) Pass through embedding + transformer
        x = self.embedding_layer(token_ids_padded)
        x = self.transformer_block(x, training=training)
        x = self.global_pool(x)
        x = self.dense1(x)
        output = self.classifier(x)
        return output

def my_model_function():
    # Return an instance of MyModel with default parameters.
    return MyModel()

def GetInput():
    # Return a batch of example string input compatible with MyModel.
    # Producing a tf.Tensor of strings (batch,)
    example_sentences = [
        "word1 word2 word3",
        "word999 word4 word5 word6 word7",
        "word100 word200 word300",
        "unknownword word2 word3",  # some OOV tokens
        ""
    ]
    # Convert to tf.Tensor shape=(batch,)
    return tf.constant(example_sentences)


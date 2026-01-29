# tf.random.uniform((B, max_len, max_len_word), dtype=tf.int32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, n_words=10000, max_len=10, max_len_word=15, n_tags=17, embedding_dim=200, embedding_matrix=None):
        super().__init__()
        # Assumptions made based on the provided issue:
        # n_words: vocabulary size (plus 2 for padding/OOV)
        # max_len: max number of tokens per sequence
        # max_len_word: presumably word-level length (or subword tokens)
        # n_tags: number of output tags/classes
        # embedding_dim: dimension of embeddings
        # embedding_matrix: pre-trained embedding weights matrix, shape (n_words+2, embedding_dim)
        #
        # We replicate the architecture:
        # Input shape: (batch_size, max_len, max_len_word) -> sequences of sequences of word tokens
        # Embedding per word with mask_zero=True (to enable masking)
        # TimeDistributed over max_len dimension for word embedding and word-LSTM
        # Then Bidirectional LSTM over sentence embeddings
        # Then TimeDistributed Dense + softmax for tag prediction
        
        if embedding_matrix is None:
            # If no embedding provided, use random initialization for demo (not trainable)
            embedding_matrix = tf.random.uniform(shape=(n_words + 2, embedding_dim), dtype=tf.float32)
        
        self.embedding = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Embedding(
                input_dim=n_words + 2,
                output_dim=embedding_dim,
                input_length=max_len_word,
                weights=[embedding_matrix],
                trainable=False,
                mask_zero=True  # important to replicate issue context & masking
            )
        )
        
        self.word_lstm = tf.keras.layers.TimeDistributed(
            tf.keras.layers.LSTM(units=32, return_sequences=False)
        )
        
        self.main_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=64, return_sequences=True)
        )
        
        self.classifier = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(n_tags + 1, activation="softmax")  # +1 for an extra tag presumably
        )
        
    def call(self, inputs, training=False):
        """
        Forward pass:
        inputs: (batch_size, max_len, max_len_word), int32 indices
        
        Steps:
        - Pass input through embedding: output shape (batch, max_len, max_len_word, embedding_dim)
        - Pass through word-level LSTM per TimeDistributed: output shape (batch, max_len, 32)
        - Pass through bidirectional main LSTM: output shape (batch, max_len, 128) [64*2]
        - Pass through TimeDistributed dense with softmax for tag prediction:
            output shape (batch, max_len, n_tags+1)
        """
        x = self.embedding(inputs)  # shape: (B, max_len, max_len_word, embedding_dim)
        x = self.word_lstm(x)       # shape: (B, max_len, 32)
        x = self.main_lstm(x)       # shape: (B, max_len, 128)
        out = self.classifier(x)    # shape: (B, max_len, n_tags+1)
        return out

def my_model_function():
    # For demonstration, we will set sensible default values assumed from the issue context.
    n_words = 10000      # example vocabulary size
    max_len = 10         # max sequence length 
    max_len_word = 15    # max word length (tokens)
    n_tags = 17          # number of tags to predict
    embedding_dim = 200  # embedding dimension
    
    # For the embedding matrix - normally comes from pretrained embeddings like GloVe
    embedding_matrix = tf.random.uniform(shape=(n_words + 2, embedding_dim), dtype=tf.float32)
    
    model = MyModel(
        n_words=n_words,
        max_len=max_len,
        max_len_word=max_len_word,
        n_tags=n_tags,
        embedding_dim=embedding_dim,
        embedding_matrix=embedding_matrix
    )
    return model

def GetInput():
    # Return random input tensor matching (batch_size, max_len, max_len_word)
    # Values are integer indices for embedding lookup.
    batch_size = 4       # small batch for testing
    max_len = 10         # must match model's max_len
    max_len_word = 15    # must match model's max_len_word
    n_words = 10000      # vocab size assumed
    # Generate random ints in the vocab index range [0, n_words+1]
    # Including zero because mask_zero=True is used and zero is reserved for padding
    input_tensor = tf.random.uniform(
        shape=(batch_size, max_len, max_len_word),
        minval=0,
        maxval=n_words + 2,
        dtype=tf.int32
    )
    return input_tensor


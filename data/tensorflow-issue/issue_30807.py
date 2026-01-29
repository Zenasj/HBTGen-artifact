# tf.random.uniform((B,), dtype=tf.int32) ← inferred input feature 'x' of shape (batch_size, 10) as integer indices

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # In the original code, an untrainable large variable embedding of shape (3,000,000, 300) was created on CPU
        # Since this is very large, here we simulate a smaller example for illustration.
        # We'll keep the interface compatible with large embedding lookup.
        self.embedding_dim = 300
        self.vocab_size = 3000000  # 3 million
        # Create an untrainable variable initialized from a random normal normal distribution
        # Note: dtype is float64 in original but using float32 for practical reasons
        # Using trainable=False as in original
        # In a real scenario, loading/feeding this large variable may cause memory issue as per the report.
        self.big_embedding = tf.Variable(
            tf.random.normal([self.vocab_size, self.embedding_dim], dtype=tf.float32),
            trainable=False, name="big"
        )
        # Dense projection layer from embedding dimension to logits (2 classes)
        self.dense = tf.keras.layers.Dense(2)

    @tf.function(jit_compile=True)
    def call(self, features):
        # features is expected to be dict with key 'x', a tensor of shape (B, 10)
        seq = features['x']  # shape: (batch_size, 10)
        # embedding lookup: shape (batch_size, 10, embedding_dim)
        emb = tf.nn.embedding_lookup(self.big_embedding, seq)
        # Apply dense layer to embeddings
        logits = self.dense(emb)  # shape (batch_size, 10, 2)
        # For simplicity, reduce mean logits over sequence length and classes - mimics the 'loss' in original code
        # This is not a full loss; just a sample output as per the original model setup
        output = tf.reduce_mean(logits)
        return output


def my_model_function():
    # Returns an instance of MyModel
    return MyModel()


def GetInput():
    # Returns a dummy input dict matching expected input features, i.e. {'x': Tensor of shape (B, 10) with integer indices}
    # Choose batch size 4 arbitrarily
    batch_size = 4
    seq_length = 10
    # Indices must be integers in [0, vocab_size)
    # Using tf.random.uniform to generate random indices
    x = tf.random.uniform(
        shape=(batch_size, seq_length),
        minval=0,
        maxval=3000000,
        dtype=tf.int32
    )
    return {'x': x}


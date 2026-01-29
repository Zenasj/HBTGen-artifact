# tf.random.uniform((B, S), dtype=tf.int64) ← input is a batch of sequences of token indices (word indices)
import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A Keras model that uses a pretrained, non-trainable embedding layer followed by a simple sequence model.
    
    This example incorporates the pattern discussed in the issue:
    - Use tf.keras.layers.Embedding initialized with pretrained embeddings
    - Freeze the embeddings (trainable=False)
    - Follow with a simple sequence model (e.g., LSTM) for downstream task
    
    Assumptions:
    - Input shape is (batch_size, sequence_length), dtype int64 (token indices)
    - pretrained_embeddings is a 2D numpy array or tensor of shape (vocab_size, embedding_dim)
    - The downstream architecture here uses an LSTM and Dense layer as a placeholder
      for "any model architecture" after the embeddings.
    """

    def __init__(self, pretrained_embeddings, **kwargs):
        super().__init__(**kwargs)
        vocab_size, embedding_dim = pretrained_embeddings.shape
        
        # Embedding layer: loaded with pretrained embeddings and frozen
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            trainable=False,
            name="pretrained_embedding"
        )
        # Must build embedding to set weights
        self.embedding.build((None,))
        self.embedding.set_weights([pretrained_embeddings])
        
        # Example downstream sequence model:
        self.lstm = tf.keras.layers.LSTM(units=64, name="lstm_layer")
        self.output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid', name="output_layer")
        
    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.lstm(x)
        output = self.output_layer(x)
        return output


def my_model_function():
    """
    Instantiate MyModel with dummy pretrained embeddings.
    
    For demonstration, we create a random embedding matrix.
    Replace `embedding_matrix` with actual pretrained embeddings.
    """
    # Assume vocab size 10000 tokens, embedding dim 300
    vocab_size = 10000
    embedding_dim = 300
    
    # Random embeddings for demonstration; in practice load real pretrained embeddings
    embedding_matrix = tf.random.uniform((vocab_size, embedding_dim), dtype=tf.float32)
    
    return MyModel(pretrained_embeddings=embedding_matrix)


def GetInput():
    """
    Returns a random input tensor of shape (batch_size, sequence_length) with int64 token indices.
    Matches the input expected by MyModel.
    """
    batch_size = 4
    sequence_length = 20
    vocab_size = 10000  # must match embedding vocab size

    # Random integers in valid token index range
    return tf.random.uniform(
        (batch_size, sequence_length),
        minval=0,
        maxval=vocab_size,
        dtype=tf.int64
    )


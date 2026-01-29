# tf.random.uniform((B, None), dtype=tf.int32) ← Input shape: batch size, variable length sequence of token IDs (ints)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        max_tokens = 6
        dimension = 2
        
        # Embedding layer mapping token IDs to dense vectors
        self.token_embedding = tf.keras.layers.Embedding(max_tokens, dimension)
        
        # Conv1D CNN layer with 100 filters, kernel size 4 and 'same' padding
        self.cnn_layer = tf.keras.layers.Conv1D(
            filters=100,
            kernel_size=4,
            padding='same')
        
        # AdditiveAttention layer (original issue is missing get_config)
        # Here we add a proper get_config override for it
        class CustomAdditiveAttention(tf.keras.layers.AdditiveAttention):
            def get_config(self):
                config = super().get_config()
                return config
        self.attention = CustomAdditiveAttention()
        
        # Global average pooling to reduce sequence dimension
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        
        # Concatenate layer: implemented in call by tf.concat
        
    def call(self, inputs):
        # inputs: tuple of (query_input, value_input)
        query_input, value_input = inputs  # Both are integer token ID sequences: shape (batch, sequence_length)
        
        # Embed tokens
        query_embeddings = self.token_embedding(query_input)  # (batch, Tq, dimension)
        value_embeddings = self.token_embedding(value_input)  # (batch, Tv, dimension)
        
        # CNN encoding
        query_seq_encoding = self.cnn_layer(query_embeddings)  # (batch, Tq, filters)
        value_seq_encoding = self.cnn_layer(value_embeddings)  # (batch, Tv, filters)
        
        # Additive attention between query and value sequences
        query_value_attention_seq = self.attention([query_seq_encoding, value_seq_encoding])  # (batch, Tq, filters)
        
        # Global average pooling over sequence dimension
        query_encoding = self.global_avg_pool(query_seq_encoding)  # (batch, filters)
        query_value_attention = self.global_avg_pool(query_value_attention_seq)  # (batch, filters)
        
        # Concatenate pooled query and attention encodings
        output = tf.concat([query_encoding, query_value_attention], axis=-1)  # (batch, filters*2)
        return output
    
    def get_config(self):
        # Provide config so the model can be serialized if needed
        # Note: since this class is a Model subclass, get_config is optional but good practice.
        # We only store basic configs here as layers' configs are handled automatically.
        config = super().get_config()
        return config

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Create a sample input matching the expected input for MyModel
    
    # Assumptions:
    # - batch size = 4 (arbitrary)
    # - sequence length = variable (use 8)
    # - token IDs range from 0 to max_tokens-1 (max_tokens=6)
    batch_size = 4
    seq_len = 8
    max_tokens = 6
    
    # Generate two input tensors of shape (batch_size, seq_len) with int32 token IDs
    query_input = tf.random.uniform((batch_size, seq_len), minval=0, maxval=max_tokens, dtype=tf.int32)
    value_input = tf.random.uniform((batch_size, seq_len), minval=0, maxval=max_tokens, dtype=tf.int32)
    
    return (query_input, value_input)


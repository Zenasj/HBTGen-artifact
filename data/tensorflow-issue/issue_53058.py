# tf.random.uniform((B, T, ...), dtype=tf.float32) ← Assumption: input shape typically (batch_size, sequence_length, features) for attention layers

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, 
                 causal=False, 
                 dropout=0.0, 
                 return_attention_scores=False, 
                 **kwargs):
        super().__init__(**kwargs)
        self.causal = causal
        self.dropout = dropout
        # This parameter controls whether attention scores are returned
        self.return_attention_scores = return_attention_scores
        
        # For demonstration, implement a simple scaled dot-product attention block
        # In real usage, this would be the BaseDenseAttention-derived layer
        
        # Simple dense layers to simulate queries, keys, values
        self.query_dense = tf.keras.layers.Dense(64)
        self.key_dense = tf.keras.layers.Dense(64)
        self.value_dense = tf.keras.layers.Dense(64)
        
        # Dropout layer, applied to attention weights (if dropout > 0)
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        
    def call(self, inputs, mask=None, training=None):
        """
        inputs: Tensor of shape (batch_size, seq_len, features)
        mask: Optional mask tensor (batch_size, seq_len) or None
        training: Boolean or None

        Returns:
            - if return_attention_scores==False: output tensor of shape (batch_size, seq_len, features)
            - if True: tuple (output, attention_scores) where attention_scores shape (batch_size, seq_len, seq_len)
        """
        # Compute queries, keys, values
        queries = self.query_dense(inputs)  # (B, T, D)
        keys = self.key_dense(inputs)       # (B, T, D)
        values = self.value_dense(inputs)   # (B, T, D)
        
        # Scaled dot-product attention
        depth = tf.cast(tf.shape(keys)[-1], tf.float32)
        logits = tf.matmul(queries, keys, transpose_b=True)  # (B, T, T)
        logits /= tf.math.sqrt(depth)
        
        if mask is not None:
            # Assumes mask shape broadcastable to (B, T, T)
            mask = tf.cast(mask, logits.dtype)
            mask = tf.expand_dims(mask, axis=1)  # (B, 1, T) to broadcast 
            logits += (1.0 - mask) * -1e9
        
        attention_weights = tf.nn.softmax(logits, axis=-1)  # (B, T, T)
        
        # Apply dropout to attention weights during training
        attention_weights = self.dropout_layer(attention_weights, training=training)
        
        output = tf.matmul(attention_weights, values)  # (B, T, D)
        
        if self.return_attention_scores:
            return output, attention_weights
        else:
            return output

    def compute_output_shape(self, input_shape):
        """
        input_shape: Tuple or tf.TensorShape (batch_size, seq_len, features)
        
        Returns:
            Output shape(s) depending on self.return_attention_scores
        """
        batch_size, seq_len, features = input_shape
        
        output_shape = tf.TensorShape((batch_size, seq_len, 64))  # 64 is model output dim from value_dense
        
        if self.return_attention_scores:
            # output + attention weights shape
            attention_shape = tf.TensorShape((batch_size, seq_len, seq_len))
            return output_shape, attention_shape
        else:
            return output_shape

def my_model_function():
    # Return an instance with return_attention_scores defaulting to False
    return MyModel()

def GetInput():
    # Assumed input is float tensor with shape (batch_size=2, sequence_length=10, features=32)
    # This is a typical shape for attention layers.
    return tf.random.uniform((2, 10, 32), dtype=tf.float32)


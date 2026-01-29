# tf.random.uniform((B, 30, 3), dtype=tf.float32), tf.random.uniform((B, 30, max_sequence_length), dtype=tf.int32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim=128, max_sequence_length=50):
        super().__init__()
        # LSTM for numeric inputs (shape: batch, 30 days, 3 features)
        self.num_lstm = tf.keras.layers.LSTM(64)
        self.num_dense = tf.keras.layers.Dense(64)
        
        # Embedding + LSTM for text inputs (shape: batch, 30 days, max_sequence_length)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.text_lstm = tf.keras.layers.LSTM(64)
        
        # Final dense layers after concatenation
        self.concat_dense = tf.keras.layers.Dense(64)
        self.output_dense = tf.keras.layers.Dense(2, name="prediction")
    
    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        # inputs is a tuple: (nums, text)
        nums, text = inputs
        
        # Numeric pipeline: nums shape (B,30,3)
        num_lstm_out = self.num_lstm(nums, training=training)  # output shape (B, 64)
        num_dense_out = self.num_dense(num_lstm_out)           # output shape (B, 64)
        
        # Text pipeline: text shape (B,30,max_sequence_length), but LSTM expects 3D with (batch, timesteps, features)
        # We need to embed each day's sequence of words: shape becomes (B,30, max_seq_len) -> embedding produces (B,30,max_seq_len,embedding_dim)
        # LSTM expects 3D, so we must reshape or merge the day and sequence dims for the LSTM.
        # But goal is to keep time dimension 30, so model should process each day as a sequence of words.
        # The original code error: "expected ndim=3, found ndim=4" - embedding output is 4D.
        # Solution: treat the 30 as timesteps and max_sequence_length as features (words embedded)
        # To do this, we can reshape the embedded input so that input to LSTM has shape (B,30, embedding_dim*max_sequence_length)
        # or better: merge last two dims - but embedding outputs (B,30,max_seq_len, embedding_dim)
        # Reshape to (B,30,max_seq_len*embedding_dim)
        
        emb = self.embedding(text)  # (B,30,max_seq_len,embedding_dim)
        B = tf.shape(emb)[0]
        emb_reshaped = tf.reshape(emb, (B, 30, -1))  # (B,30, max_seq_len*embedding_dim)
        
        text_lstm_out = self.text_lstm(emb_reshaped, training=training) # (B,64)
        
        # Concatenate numeric and text features
        united = tf.concat([num_dense_out, text_lstm_out], axis=-1)  # (B,128)
        
        almostlast = self.concat_dense(united)  # (B,64)
        last = self.output_dense(almostlast)    # (B,2)
        
        return last

def my_model_function():
    # Provide dummy vocab_size and max_sequence_length for initialization,
    # These must be set appropriately in practice.
    vocab_size = 10000  # Example vocab size
    max_sequence_length = 50  # Example max sequence length of text sequences per day
    
    return MyModel(vocab_size=vocab_size, embedding_dim=128, max_sequence_length=max_sequence_length)

def GetInput():
    # Generate dummy input matching the expected inputs of the model
    # Based on user data:
    # Numeric input: shape (batch, 30 days, 3 features)
    # Text input: shape (batch, 30 days, max_sequence_length), integer sequences (word indices)
    
    batch_size = 4
    num_days = 30
    num_features = 3
    max_sequence_length = 50
    vocab_size = 10000  # same as model
    
    numeric_input = tf.random.uniform((batch_size, num_days, num_features), dtype=tf.float32, minval=0, maxval=100)
    text_input = tf.random.uniform((batch_size, num_days, max_sequence_length), dtype=tf.int32, minval=1, maxval=vocab_size)
    
    return (numeric_input, text_input)


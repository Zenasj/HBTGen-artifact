# tf.random.uniform((B, T), dtype=tf.string) ← B=batch size, T=sequence length (variable)

import tensorflow as tf
import tensorflow_hub as hub

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Load ELMo from TF Hub
        self.embed = hub.KerasLayer(
            "https://tfhub.dev/google/elmo/2",
            trainable=True,
            signature="default",
            output_key="elmo",
        )
        # ELMo embedding dimension is 1024
        self.elmo_dim = 1024
        
        # RNN and dense layers as per original model
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)
        )
        self.dense_relu = tf.keras.layers.Dense(32, activation='relu')
        self.dense_out = tf.keras.layers.Dense(2, activation='sigmoid')
    
    def call(self, inputs, training=False):
        # inputs shape: (batch_size, sequence_length) of dtype string
        
        # Cast inputs to string tensor explicitly
        x_str = tf.cast(inputs, tf.string)
        
        # Important fix: squeeze only axis=1 to remove the sequence length dimension if size=1,
        # to avoid empty shape which causes error in StringSplit inside ELMo module
        # If sequence length >1, no squeeze happens and shape stays (B, T)
        # If T=1, shape becomes (B,) which ELMo expects for a batch of sentences.
        x_squeezed = tf.squeeze(x_str, axis=1)
        
        # Pass through ELMo embedding module
        # The hub.KerasLayer with signature='default' and output_key='elmo' returns (B, T, 1024)
        # With T squeezed away if =1, it returns (B, 1024)
        elmo_emb = self.embed(x_squeezed)
        
        # If original input had sequence length >1, elmo_emb shape is (B, T, 1024)
        # If T=1 and squeezed, elmo_emb shape is (B, 1024)
        # We want a consistent shape for BiLSTM input: (B, T, 1024)
        # So add back time dimension if missing:
        if len(elmo_emb.shape) == 2:
            elmo_emb = tf.expand_dims(elmo_emb, axis=1)  # shape (B, 1, 1024)
        
        x = self.bilstm(elmo_emb, training=training)  # output shape: (B, 256)
        x = self.dense_relu(x)
        output = self.dense_out(x)
        
        return output

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random tensor input matching shape (batch_size, sequence_length) with dtype string
    # To test single or multiple inputs:
    # We'll produce batch_size=2, sequence_length=1 example
    sample_texts = [
        ["hello"],               # batch element 1 with single string
        ["my name is Simone"],   # batch element 2 with single string
    ]
    # Convert to tf.Tensor of shape (2,1) dtype string
    return tf.constant(sample_texts, dtype=tf.string)


# tf.random.uniform((B, None), dtype=tf.int64) ← Inferred: Inputs are integer sequences with variable length (None timestep dimension)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define layers to process each timestep individually and support multiple inputs
        # Since TimeDistributed does not support multiple inputs directly,
        # we implement custom processing splitting along timestep dimension and concatenating inputs.
        
        # We will simulate a "time-distributed" concat of two integer input sequences at each timestep,
        # followed by a Dense layer for demonstration.
        
        # Embedding layer to embed integer inputs (e.g. vocab_size=10, embedding_dim=4)
        self.embedding1 = tf.keras.layers.Embedding(input_dim=10, output_dim=4)
        self.embedding2 = tf.keras.layers.Embedding(input_dim=10, output_dim=4)
        
        # Dense layer to process concatenated embeddings at each timestep
        # Applied to concatenated embeddings (so input_dim=8)
        self.dense = tf.keras.layers.Dense(8, activation='relu')
        
    def call(self, inputs, training=None):
        # inputs is a list or tuple of two tensors: [input_1, input_2]
        input_1, input_2 = inputs
        
        # Shapes:
        # input_1: (batch_size, timesteps)
        # input_2: (batch_size, timesteps)
        
        # Embed each sequence
        emb_1 = self.embedding1(input_1)  # (batch_size, timesteps, embed_dim=4)
        emb_2 = self.embedding2(input_2)  # (batch_size, timesteps, embed_dim=4)
        
        # Concatenate embeddings along last dimension => (batch_size, timesteps, 8)
        combined = tf.concat([emb_1, emb_2], axis=-1)
        
        # Apply dense layer to the combined embeddings at every timestep
        # This simulates TimeDistributed(Dense(...)) behavior
        
        # Dense layer applied to 3D tensor: treats last dimension as input_dim
        output = self.dense(combined)  # (batch_size, timesteps, 8)
        
        return output


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return example input tensors matching MyModel inputs:
    # Two integer sequences of shape (batch_size, timesteps)
    
    batch_size = 1
    timesteps = 7  # Example fixed length sequences (can be variable length as None)
    
    # Generate integer sequences in range [0, 9]
    input_1 = tf.random.uniform((batch_size, timesteps), minval=0, maxval=10, dtype=tf.int64)
    input_2 = tf.random.uniform((batch_size, timesteps), minval=0, maxval=10, dtype=tf.int64)
    
    return [input_1, input_2]


# tf.random.uniform((1, 13), dtype=tf.int32) for "word" input (shape: [1, 13])
# tf.random.uniform((1, 1), dtype=tf.int32) for "punctuation" input (shape: [1, 1])
# tf.random.uniform((1, 1), dtype=tf.int32) for "capitalisation" input (shape: [1, 1])
# tf.random.uniform((9, 1, 8, 128), dtype=tf.float32) for "input_cache" (shape: [9, 1, 8, 128])

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Parameters inferred from the config.pbtxt and usage
        self.MAX_FORWARD = 4
        self.MAX_BACKWARD = 8
        self.LAYER_SIZE = 128
        self.NUM_HIDDEN_LAYERS_MINUS_ONE = 9
        self.BATCH_SIZE = 1  # fixed batch size for cache dims
        
        # For this example, we'll build a simplified Transformer sub-module plus output heads
        # The original system runs a complicated XLA compiled model, 
        # here we mock the logic with standard layers matching shapes as best inferred.
        
        # For the input "word" which is a sequence of length 13 (1 + MAX_FORWARD + MAX_BACKWARD)
        # We'll embed and process via a simple dense layer for illustration.
        
        self.word_embedding = tf.keras.layers.Embedding(
            input_dim=10000, output_dim=64,  # vocab size assumed
            input_length=1 + self.MAX_FORWARD + self.MAX_BACKWARD)
        
        # Simulate transformer by a stack of simple Dense + normalization layers
        self.transformer_layers = [
            tf.keras.layers.Dense(self.LAYER_SIZE, activation='relu')
            for _ in range(self.NUM_HIDDEN_LAYERS_MINUS_ONE + 1)
        ]
        
        # Output heads for capitalization and punctuation, 5 time delays each:
        # 5 delays, each with "cap/predicted" and "punc/predicted"
        # We'll model output shape as (5, ?) as a batch dimension is 1 so dummy dimension
        
        self.cap_output_heads = [
            tf.keras.layers.Dense(1, activation='sigmoid', name=f"cap_predicted_{i}")
            for i in range(5)
        ]
        self.punc_output_heads = [
            tf.keras.layers.Dense(1, activation='softmax', name=f"punc_predicted_{i}")
            for i in range(5)
        ]
        
        # A layer to update the cache (simulate output_cache)
        self.cache_update_layer = tf.keras.layers.Dense(self.LAYER_SIZE)
        
    def call(self, inputs):
        # inputs is a tuple of (word, punctuation, capitalisation, input_cache)
        # word: int tensor shape [1, 13]
        # punctuation: int tensor shape [1, 1]
        # capitalisation: int tensor shape [1, 1]
        # input_cache: float tensor shape [9, 1, 8, 128]
        
        word, punctuation, capitalisation, input_cache = inputs
        
        # Embed the words (remove batch dim for embedding input)
        # word shape: (1, 13) -> embed -> (1, 13, 64)
        x = self.word_embedding(tf.squeeze(word, axis=0))  # Remove batch dim for Embedding layer input: (13,)
        # Now x shape: (13, 64)
        
        # Flatten sequence for transformer simulation: (13, 64) -> (13*64,)
        x = tf.reshape(x, [1, -1])
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)  # shape: (1, LAYER_SIZE)
        
        # Incorporate input_cache by flattening and concatenation (simplified)
        cache_flat = tf.reshape(input_cache, [1, -1])  # shape: (1, 9*1*8*128)
        
        combined = tf.concat([x, cache_flat], axis=1)  # concat features
        
        # Update cache (simulate output_cache)
        updated_cache = self.cache_update_layer(combined)  # shape (1, LAYER_SIZE)
        
        # Create outputs for each delay, split combined features again simplistically
        cap_outputs = []
        punc_outputs = []
        
        # For each delay i = 0..4 create separate predicted outputs
        for i in range(5):
            # Slice some features (dummy split) for each output head
            cap_out = self.cap_output_heads[i](combined)
            punc_out = self.punc_output_heads[i](combined)
            cap_outputs.append(cap_out)
            punc_outputs.append(punc_out)
        
        # Stack outputs so shape matches expected outputs: (5, 1, 1)
        cap_outputs = tf.stack(cap_outputs, axis=0)  # (5, 1, 1)
        punc_outputs = tf.stack(punc_outputs, axis=0)  # (5, 1, 1)
        
        # Reshape updated_cache to (9,1,8,128) to simulate output_cache for next step feed
        # Here we replicate updated_cache across cache dim as a dummy placeholder
        output_cache = tf.reshape(
            tf.tile(updated_cache, multiples=[self.NUM_HIDDEN_LAYERS_MINUS_ONE * 8]),
            [self.NUM_HIDDEN_LAYERS_MINUS_ONE, 1, self.MAX_BACKWARD, self.LAYER_SIZE])
        
        # Return outputs as dictionary to match fetch nodes semantics in original config
        return {
            "output_cache": output_cache,
            "cap_predicted": cap_outputs,   # shape (5,1,1)
            "punc_predicted": punc_outputs  # shape (5,1,1)
        }


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a tuple of inputs matching the expected inputs of MyModel
    
    import numpy as np
    
    # word input: int32 shape [1, 13], values simulating word indices 0..9999
    word = tf.constant(np.random.randint(0, 10000, size=(1, 13)), dtype=tf.int32)
    
    # punctuation input: int32 shape [1,1], dummy 0 or 1
    punctuation = tf.constant(np.random.randint(0, 2, size=(1,1)), dtype=tf.int32)
    
    # capitalisation input: int32 shape [1,1], dummy 0 or 1
    capitalisation = tf.constant(np.random.randint(0, 2, size=(1,1)), dtype=tf.int32)
    
    # input_cache: float32 shape [9,1,8,128], filled zeros initially
    input_cache = tf.zeros((9, 1, 8, 128), dtype=tf.float32)
    
    return (word, punctuation, capitalisation, input_cache)


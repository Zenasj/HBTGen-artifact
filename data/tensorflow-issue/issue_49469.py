# tf.random.uniform((B, S), dtype=tf.int32) where B=batch_size, S=sequence_length

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size=1000):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = tf.keras.layers.Embedding(vocab_size, 32)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training=None):
        """
        Forward pass implementing a loop over the sequence length dimension,
        applying a Dense layer to each token embedding separately,
        and collecting outputs via a tf.TensorArray.
        
        This was the original pattern from the issue that caused TPU/XLA compilation issues
        when using tf.range inside loops with TensorArray.
        
        The method stacks outputs for each token and returns a transposed output
        tensor of shape (batch_size, sequence_length, vocab_size).
        
        Note: token_length is determined dynamically from input shape.
        """
        embedding = self.embed(inputs)
        token_length = tf.shape(embedding)[1]  # dynamic shape to avoid static shape issues
        
        outputs = tf.TensorArray(tf.float32, size=token_length)
        
        # Use tf.while_loop for TPU/XLA compatibility instead of python range iteration
        i = tf.constant(0)
        
        def cond(i, outputs):
            return i < token_length
        
        def body(i, outputs):
            # Process embedding for token at position i
            output = self.dense(embedding[:, i, :])
            outputs = outputs.write(i, output)
            return i + 1, outputs
        
        _, outputs = tf.while_loop(cond, body, [i, outputs], maximum_iterations=token_length)
        
        # outputs.stack() returns shape (token_length, batch_size, vocab_size),
        # so transpose to (batch_size, token_length, vocab_size)
        return tf.transpose(outputs.stack(), perm=[1, 0, 2])

def my_model_function():
    # Instantiate MyModel with vocab_size 1000 as in the original issue
    return MyModel(vocab_size=1000)

def GetInput():
    # Generate a random int32 tensor to simulate input token IDs as expected by embedding layer.
    # Shape: (batch_size=32, sequence_length=32)
    batch_size = 32
    sequence_length = 32
    vocab_size = 1000  # must match model's vocab_size
    
    # Generate integer token IDs between 0 and vocab_size-1
    return tf.random.uniform(
        shape=(batch_size, sequence_length),
        minval=0,
        maxval=vocab_size,
        dtype=tf.int32
    )


# tf.random.uniform((B,), dtype=tf.int32) ← Input is a batch of token indices (1D integer tensor)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding layer with fixed weights initialized from a known constant matrix
        # Using the same shape and initialization pattern as original issue:
        # input_dim=10, output_dim=5, trainable=False
        embedding_matrix = tf.random.uniform((10, 5), dtype=tf.float32)
        self.embedding = tf.keras.layers.Embedding(
            input_dim=10,
            output_dim=5,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
            trainable=False,
        )
        # Store the embedding_matrix explicitly for comparison in call
        self.embedding_matrix = embedding_matrix

    @tf.function(jit_compile=True)
    def call(self, inputs):
        """
        inputs: A 1D int32 tensor containing indices 
        Returns: A boolean tensor indicating elementwise equality 
                 between embeddings from the layer and embeddings directly 
                 taken from embedding_matrix via gather.
        """
        embedded = self.embedding(inputs)  # Shape: (batch_size, embedding_dim)
        # Grab expected embedding vectors directly from stored matrix
        expected = tf.gather(self.embedding_matrix, inputs)
        # Compare with a tolerance to handle float issues (tf.equal is strict)
        # Use tf.math.abs diff < small epsilon, elementwise
        diff = tf.math.abs(embedded - expected)
        is_equal = tf.reduce_all(diff < 1e-6, axis=-1)  # shape: (batch_size,)

        # For demonstration, return the boolean vector comparing embeddings per input
        return is_equal

def my_model_function():
    # Return an instance of MyModel with random embedding matrix initialized internally
    return MyModel()

def GetInput():
    # Return a random batch of integers between 0 and 9 inclusive, shape (B,)
    # Assuming batch size 3 arbitrarily (can be changed as needed)
    B = 3
    return tf.random.uniform((B,), minval=0, maxval=10, dtype=tf.int32)


# tf.random.uniform((B, 1), dtype=tf.int32) ← input shape is (batch_size, 1) with categorical indices for embeddings
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, embedding_size_bytes: int = 20 * 1024 * 1024 * 1024, var_count: int = 1):
        # embedding_size_bytes default to 20GB as in the example
        super().__init__()
        self.embedding_size_bytes = embedding_size_bytes
        self.var_count = var_count

        # Define a Dense layer, kernel initialized with GlorotNormal seed 0, sigmoid activation
        self.dense = tf.keras.layers.Dense(
            1, activation="sigmoid",
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=0)
        )

        # Compute vocab size by dividing embedding_size_bytes by (256 * 4) 
        # Because each embedding is [vocab_size, 256] of float32 (4 bytes)
        self.vocab_size = int(self.embedding_size_bytes / (256 * 4))

        # Store embeddings as dict of variables, created on build
        self.embeddings = {}

    def build(self, input_shape):
        # Create one or more embedding variables (sharded in PS scenario)
        for i in range(self.var_count):
            name = f"embedding_{i}"
            self.embeddings[name] = self.add_weight(
                name=name,
                shape=(self.vocab_size, 256),
                initializer=tf.keras.initializers.RandomUniform(seed=i),
                dtype=tf.float32,
                trainable=True,
            )
        # Build dense with input shape [batch_size, 256]
        self.dense.build(tf.TensorShape((None, 256)))
        self.built = True

    def call(self, inputs):
        # inputs shape assumed [batch_size, 1], int indices into embeddings
        # For each embedding variable, lookup the embeddings using tf.nn.embedding_lookup
        embedding_values = []
        for i in range(self.var_count):
            name = f"embedding_{i}"
            # Squeeze axis 1 to get shape [batch_size]
            indices = tf.squeeze(inputs, axis=1)
            emb = tf.nn.embedding_lookup(self.embeddings[name], indices)
            embedding_values.append(emb)
        # Average embeddings if multiple variables
        overall_embedding = tf.reduce_mean(embedding_values, axis=0)
        # Pass through dense layer to produce output with sigmoid activation
        output = self.dense(overall_embedding)
        return output


def my_model_function():
    # Instantiate MyModel with default 20GB embedding, 1 var_count as in example
    model = MyModel(embedding_size_bytes=20 * 1024 * 1024 * 1024, var_count=1)
    # Build model for fixed input shape (None, 1)
    model.build(tf.TensorShape((None, 1)))
    return model


def GetInput():
    # Generate a batch of random indices as input tensor to MyModel
    # Batch size chosen as 4 arbitrarily, input shape [4, 1]
    batch_size = 4
    vocab_size = int(20 * 1024 * 1024 * 1024 / (256 * 4))
    # Random indices in [0, vocab_size) as int32 tensor
    inputs = tf.random.uniform(
        shape=(batch_size, 1),
        minval=0,
        maxval=vocab_size,
        dtype=tf.int32
    )
    return inputs


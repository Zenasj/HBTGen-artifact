# tf.random.uniform((batch_size, None), dtype=tf.string)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define two separate categorical columns with distinct hash buckets
        self.feature_alpha = tf.feature_column.categorical_column_with_hash_bucket(
            "feature_alpha", hash_bucket_size=100, dtype=tf.string
        )
        self.feature_beta = tf.feature_column.categorical_column_with_hash_bucket(
            "feature_beta", hash_bucket_size=200, dtype=tf.string
        )

        # Define separate embedding columns with distinct dimensions
        self.alpha_emb = tf.feature_column.embedding_column(self.feature_alpha, dimension=10)
        self.beta_emb = tf.feature_column.embedding_column(self.feature_beta, dimension=20)

        # Define two separate DenseFeatures layers — as a workaround to avoid variable name conflicts in TF 2.3.0
        self.dense_features_alpha = tf.keras.layers.DenseFeatures([self.alpha_emb], name="dense_features_alpha")
        self.dense_features_beta = tf.keras.layers.DenseFeatures([self.beta_emb], name="dense_features_beta")

        # A dense layer after concatenating the embeddings
        self.dense = tf.keras.layers.Dense(64, activation='relu')

    def call(self, inputs):
        # Extract embeddings separately to keep variable namespaces distinct
        alpha_embedding = self.dense_features_alpha(inputs)
        beta_embedding = self.dense_features_beta(inputs)

        # Concatenate embeddings along last dimension
        concat_emb = tf.concat([alpha_embedding, beta_embedding], axis=-1)

        out = self.dense(concat_emb)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    # Use batch size 4 as an example; sequence length is variable (None)
    # Inputs are sparse string tensors shaped (batch_size, variable length)
    batch_size = 4
    # Create dummy sparse inputs (ragged tensor for simplicity)
    # The DenseFeatures layer expects SparseTensors; however, Keras Input with sparse=True works with RaggedTensors as well.
    # We'll create dense tensors with shape (batch_size, max_seq_len) to simulate sparse inputs.
    max_seq_len_alpha = 3
    max_seq_len_beta = 2

    # Random strings (hash buckets will apply)
    alpha_values = tf.random.uniform([batch_size, max_seq_len_alpha], maxval=1000, dtype=tf.int32)
    beta_values = tf.random.uniform([batch_size, max_seq_len_beta], maxval=1000, dtype=tf.int32)

    # Convert integers to strings ('str{num}')
    def int_to_string(t):
        t = tf.strings.as_string(t)
        t = tf.strings.join(["str", t])
        return t

    inputs = {
        "feature_alpha": int_to_string(alpha_values),
        "feature_beta": int_to_string(beta_values),
    }
    return inputs


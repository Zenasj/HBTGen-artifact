# tf.random.uniform((B, None), dtype=tf.string) ← Input is a dict of variable-length string sequences with keys 'test' and 'test2'

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Embedding for 'test' input vocabulary
        vocab_1 = ['This', 'That', 'Thing']
        self.lookup_1 = tf.keras.layers.StringLookup(
            vocabulary=vocab_1, mask_token=None, oov_token=None, output_mode='int')
        self.embedding_1 = tf.keras.layers.Embedding(input_dim=len(vocab_1), output_dim=4,
                                                     embeddings_initializer='uniform')

        # Embedding for 'test2' input vocabulary
        vocab_2 = ['a', 'b', 'c']
        self.lookup_2 = tf.keras.layers.StringLookup(
            vocabulary=vocab_2, mask_token=None, oov_token=None, output_mode='int')
        self.embedding_2 = tf.keras.layers.Embedding(input_dim=len(vocab_2), output_dim=4,
                                                     embeddings_initializer='uniform')

        # To combine variable length embeddings by averaging
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()

        # Dense layer after concatenation of embeddings
        self.dense_layer = tf.keras.layers.Dense(units=35, use_bias=False, name="user-embeddings-layer")

    def call(self, inputs, training=False):
        # inputs is a dict with keys 'test' and 'test2' of shape (batch_size, None),
        # dtype string
        x1 = inputs['test']   # shape (batch_size, seq_len1)
        x2 = inputs['test2']  # shape (batch_size, seq_len2)

        # Convert strings to integer indices
        x1_ids = self.lookup_1(x1)  # shape (batch_size, seq_len1)
        x2_ids = self.lookup_2(x2)  # shape (batch_size, seq_len2)

        # Embed each token id
        emb1 = self.embedding_1(x1_ids)  # (batch_size, seq_len1, 4)
        emb2 = self.embedding_2(x2_ids)  # (batch_size, seq_len2, 4)

        # Average over the sequence length dimension to get fixed vector per input
        pooled_emb1 = self.global_avg_pool(emb1)  # (batch_size, 4)
        pooled_emb2 = self.global_avg_pool(emb2)  # (batch_size, 4)

        # Concatenate embeddings from both inputs
        concat_emb = tf.keras.layers.concatenate([pooled_emb1, pooled_emb2])  # (batch_size, 8)

        # Apply final dense layer
        output = self.dense_layer(concat_emb)  # (batch_size, 35)

        return output

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile the model similarly to given example, using Adagrad and sparse categorical crossentropy
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def GetInput():
    # Generate a dict input consistent with the model's expected input:
    # Both inputs are 1D variable length string tensors (ragged or padded).
    # Use batch size 2 as in original snippet.
    input_test = tf.constant([['This', 'That'], ['That', 'Thing']])  # shape (2, 2)
    input_test2 = tf.constant([['a', 'b'], ['b', 'c']])             # shape (2, 2)
    return {'test': input_test, 'test2': input_test2}


# tf.random.uniform((2, 8), dtype=tf.int32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Parameters inferred from the issue description
        self.vocab_len = 5
        self.emb_dim = 20
        self.num_entities = 3
        # Embedding layer with TruncatedNormal initializer and mask_zero=True
        emb_ini = tf.keras.initializers.TruncatedNormal()
        self.emb = tf.keras.layers.Embedding(
            input_dim=self.vocab_len,
            output_dim=self.emb_dim,
            embeddings_initializer=emb_ini,
            mask_zero=True)
        # TimeDistributed Dense layer to output logits for num_entities classes per time step
        self.td_dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.num_entities))
        # Softmax activation per time step
        self.activation = tf.keras.layers.Activation('softmax')

    def call(self, inputs):
        x = self.emb(inputs)  # shape (batch_size, seq_len, emb_dim)
        x = self.td_dense(x)  # shape (batch_size, seq_len, num_entities)
        x = self.activation(x)  # shape (batch_size, seq_len, num_entities)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random integer tensor mimicking the batch input:
    # Batch size = 2, sequence length = 8, dtype int32 (token ids)
    # Values in range [0, vocab_len), zero for mask
    return tf.random.uniform(
        (2, 8), minval=0, maxval=5, dtype=tf.int32)


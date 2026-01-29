# tf.random.uniform((10, 5), minval=0, maxval=1000, dtype=tf.int64)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        EMBD_INPUT_DIM = 1000
        EMBD_OUTPUT_DIM = 64
        
        # Instead of relying on embedding checkpoint restore which breaks portability,
        # we initialize the embeddings manually here.
        # This mimics the original EmbeddingColumn with pre-trained embeddings,
        # but fully managed within this SavedModel.
        self.embeddings = tf.Variable(
            initial_value=tf.keras.initializers.GlorotNormal()(shape=(EMBD_INPUT_DIM, EMBD_OUTPUT_DIM)),
            trainable=True,
            name='embedding_weights'
        )
        
        # A simple dense layer to produce final output, as in original model
        self.dense = tf.keras.layers.Dense(1, name='out')
        
        # We replicate the behavior of the EmbeddingColumn + DenseFeatures with a custom embedding lookup
        # This is to avoid dependence on external ckpt files.
    
    def call(self, inputs):
        # Inputs: a dict { 'id': tf.Tensor of shape (batch, None), dtype int64 }
        ids = inputs['id']  # shape (batch, variable length)
        
        # Lookup embedding vectors for each id
        # Clip ids to max bucket to avoid invalid indices
        clipped_ids = tf.clip_by_value(ids, 0, tf.shape(self.embeddings)[0] - 1)
        
        embedded = tf.nn.embedding_lookup(params=self.embeddings, ids=clipped_ids)
        # embedded shape: (batch, id_len, embedding_dim)
        
        # Pool embeddings along id dimension - original DenseFeatures collapses variable length dimension by sum or mean.
        # The original was built on DenseFeatures with embedding_column, which sums embeddings by default.
        pooled = tf.reduce_sum(embedded, axis=1)  # shape (batch, embedding_dim)
        
        out = self.dense(pooled)  # shape (batch, 1)
        return out

def my_model_function():
    # Return an instance of MyModel with freshly initialized embeddings.
    return MyModel()

def GetInput():
    # Generate a random input dict compatible with MyModel:
    # {"id": tf.Tensor of shape (batch, id_len), int64 values in [0, 999]}
    B = 10  # test batch size
    id_len = 5  # test sequence length per example
    EMBD_INPUT_DIM = 1000
    
    input_ids = tf.random.uniform(
        shape=(B, id_len),
        minval=0,
        maxval=EMBD_INPUT_DIM,
        dtype=tf.int64
    )
    return {'id': input_ids}


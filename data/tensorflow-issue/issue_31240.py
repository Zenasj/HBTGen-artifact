# tf.random.uniform((B, None), dtype=tf.float32), tf.random.uniform((B, None), dtype=tf.int32)
import tensorflow as tf
from tensorflow.feature_column import embedding_column, sequence_categorical_column_with_identity, sequence_numeric_column
from tensorflow.keras.experimental import SequenceFeatures
from tensorflow.keras.layers import Dense, Lambda, Input
from tensorflow.keras import Model

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Sequence Numeric Column for 'denseFeat' (numeric sequence)
        self.seq_fc_dense = sequence_numeric_column('denseFeat')
        self.seq_layer_dense = SequenceFeatures(self.seq_fc_dense, name='denseFeatLayer')
        # Sequence Categorical embedding column for 'catFeat' (categorical sequence)
        nb_cat = 5
        seq_fc_cat = sequence_categorical_column_with_identity('catFeat', num_buckets=nb_cat)
        seq_fc_cat = embedding_column(seq_fc_cat, dimension=2)
        self.seq_layer_cat = SequenceFeatures(seq_fc_cat, name='catFeatLayer')

        # Output Dense layer
        self.dense_out = Dense(1, activation='sigmoid')

        # Lambda layer to convert dense numeric sequence tensor to SparseTensor as required by SequenceFeatures for numeric
        self.dense_to_sparse = Lambda(self._sparse_f)

    def _sparse_f(self, input_dense):
        # Convert dense tensor to SparseTensor required by sequence_numeric_column.
        zero = tf.constant(0, dtype=tf.float32)
        # Find indices where input_dense != 0
        indices = tf.where(tf.not_equal(input_dense, zero))
        values = tf.gather_nd(input_dense, indices)
        sparse = tf.SparseTensor(indices, values, tf.cast(tf.shape(input_dense), dtype=tf.int64))
        return sparse

    def call(self, inputs):
        # inputs is expected to be a dict {'denseFeat': dense_tensor, 'catFeat': int_tensor}
        dense_feat = inputs['denseFeat']
        cat_feat = inputs['catFeat']

        # Convert denseFeat tensor to sparse tensor before passing to SequenceFeatures for numeric column
        sparse_dense = self.dense_to_sparse(dense_feat)

        # Pass sparse input dict to corresponding SequenceFeatures layers
        x_dense = self.seq_layer_dense({'denseFeat': sparse_dense})[0]  # output is a list, take first tensor
        x_cat = self.seq_layer_cat({'catFeat': cat_feat})[0]

        # Concatenate the outputs from numeric and categorical sequence features
        x = tf.concat([x_dense, x_cat], axis=-1)
        output = self.dense_out(x)
        return output

def my_model_function():
    # Create and return an instance of the model
    return MyModel()

def GetInput():
    # Produce a tuple of inputs for 'denseFeat' (float32 tensor) and 'catFeat' (int32 tensor)
    # Assumptions:
    # - Batch size B=2 for example
    # - Variable sequence length L is simulated by random length between 3 and 6 per batch (padded with zeros)
    # - 'denseFeat' shape: (B, L) floats
    # - 'catFeat' shape: (B, L) int32 categorical IDs between 0 and 4 (since num_buckets=5)

    import numpy as np

    B = 2
    max_seq_len = 6

    # Random sequence lengths between 3 and max_seq_len to simulate variable length sequences with padding=0
    seq_lengths = np.random.randint(3, max_seq_len+1, size=B)

    dense_feat_np = np.zeros((B, max_seq_len), dtype=np.float32)
    cat_feat_np = np.zeros((B, max_seq_len), dtype=np.int32)

    for i in range(B):
        length = seq_lengths[i]
        # Random float values for denseFeat; zeros for padding beyond length
        dense_feat_np[i, :length] = np.random.uniform(low=0.1, high=1.0, size=(length,))
        # Random categorical IDs (1 to 4) for catFeat; zero-padding beyond length
        cat_feat_np[i, :length] = np.random.randint(low=1, high=5, size=(length,))

    dense_feat = tf.convert_to_tensor(dense_feat_np, dtype=tf.float32)
    cat_feat = tf.convert_to_tensor(cat_feat_np, dtype=tf.int32)

    return {'denseFeat': dense_feat, 'catFeat': cat_feat}


# tf.random.uniform((B, None, 1), dtype=tf.float32)  # Input shape inferred: batch_size, sequence_length (variable), feature_dim=1

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define sequence_numeric_column feature column
        self.sequence_numeric_col = tf.feature_column.sequence_numeric_column('a')
        self.sequence_feature_layer = tf.keras.experimental.SequenceFeatures([self.sequence_numeric_col])

        # Recurrent and output layers matching the original reported model
        self.gru = tf.keras.layers.GRU(32)
        self.dense = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        """
        inputs: dict of input tensors (string key: 'a')
        Expected input type for sequence_numeric_column: sparse tensor.
        We ensure input is a SparseTensor to avoid the TypeError "Input must be a SparseTensor."
        """
        # The SequenceFeatures layer expects sparse inputs for sequence columns.
        # However, numeric columns usually have dense inputs -- to simulate sparse,
        # we convert dense tensor input to SparseTensor here for demonstration.

        # Extract the dense input tensor from inputs dictionary
        dense_input = inputs['a']

        # If input is dense tensor, convert it to SparseTensor to satisfy SequenceFeatures requirement
        if not isinstance(dense_input, tf.SparseTensor):
            # Convert dense tensor of shape (batch, seq_len, 1) to SparseTensor
            # We need indices, values, dense_shape for SparseTensor construction.
            dense_shape = tf.shape(dense_input, out_type=tf.int64)
            # Find non-zero elements indices and values
            # For numeric sequence, zero could be valid value: so we treat all elements as present.
            # Construct indices for all elements (batch, timestep, feature_dim=1)
            batch_size = dense_shape[0]
            seq_len = dense_shape[1]
            feature_dim = dense_shape[2]

            # Build indices: meshgrid of batch, seq, feature_dim=0..0 (only one feature dim)
            b = tf.range(batch_size, dtype=tf.int64)
            s = tf.range(seq_len, dtype=tf.int64)
            f = tf.range(feature_dim, dtype=tf.int64)
            b_grid, s_grid, f_grid = tf.meshgrid(b, s, f, indexing='ij')
            b_grid = tf.reshape(b_grid, [-1])
            s_grid = tf.reshape(s_grid, [-1])
            f_grid = tf.reshape(f_grid, [-1])
            indices = tf.stack([b_grid, s_grid, f_grid], axis=1)

            values = tf.reshape(dense_input, [-1])

            sparse_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
        else:
            sparse_input = dense_input

        # Prepare input dict with expected sparse tensor for 'a'
        sparse_inputs = {'a': sparse_input}

        # Compute sequence features
        fc_layer, _ = self.sequence_feature_layer(sparse_inputs)

        x = self.gru(fc_layer)
        output = self.dense(x)
        return output

def my_model_function():
    return MyModel()

def GetInput():
    """
    Returns input dict containing a single key 'a' with:
    a sparse tensor of shape (batch_size, sequence_length, 1), dtype float32,
    matching expected input to sequence_numeric_column and model.
    
    For demonstration, batch_size=4, seq_len=5, feature_dim=1.
    SparseTensor constructed from a dense numeric tensor.
    """
    batch_size = 4
    seq_len = 5
    feature_dim = 1

    dense_values = tf.random.uniform((batch_size, seq_len, feature_dim), dtype=tf.float32)

    # Convert dense to sparse:
    dense_shape = [batch_size, seq_len, feature_dim]

    # Construct indices for all elements (simulate no missing values)
    b = tf.range(batch_size, dtype=tf.int64)
    s = tf.range(seq_len, dtype=tf.int64)
    f = tf.range(feature_dim, dtype=tf.int64)
    b_grid, s_grid, f_grid = tf.meshgrid(b, s, f, indexing='ij')
    b_grid = tf.reshape(b_grid, [-1])
    s_grid = tf.reshape(s_grid, [-1])
    f_grid = tf.reshape(f_grid, [-1])
    indices = tf.stack([b_grid, s_grid, f_grid], axis=1)

    values = tf.reshape(dense_values, [-1])

    sparse_tensor = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    
    return {'a': sparse_tensor}


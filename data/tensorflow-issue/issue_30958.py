# tf.random.uniform((B, sequence_length, C), dtype=tf.float32) with B=batch_size, sequence_length=9, C=1 (numeric), or C=embedding_units (categorical embedding)
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.experimental import SequenceFeatures
from tensorflow.python.feature_column.feature_column_v2 import EmbeddingColumn

# This model reproduces the reported issue with subclassed models using SequenceFeatures layer,
# especially the difference of handling sequence_numeric_column vs embedding_column.
# The fix applied below is: use tf.shape() (dynamic shape) when converting dense numeric inputs 
# to sparse tensor, instead of dense.shape (static shape), as the static shape has batch dimension None.

class MyModel(tf.keras.Model):
    def __init__(self, fc_list, nb_features, name='my_model', **kwargs):
        super().__init__(name=name, **kwargs)
        self.fc_list = fc_list
        self.nb_features = nb_features
        
        # Create a SequenceFeatures layer for each feature column
        self.seq_features_layers = {}
        for fc in self.fc_list:
            self.seq_features_layers[fc.name] = SequenceFeatures(fc)
        
        self.lstm = LSTM(64, return_sequences=False)
        self.output_layer = Dense(self.nb_features, activation='softmax')
    
    def call(self, inputs, training=None):
        # inputs: dict of tensors:
        # For numeric sequence column: inputs[fc_name] is dense float tensor [batch, seq_len, ...]
        # For categorical sequence column with embedding: inputs[fc_name] is integer tensor of IDs
        processed_features = {}
        
        for fc in self.fc_list:
            fc_name = fc.name
            # Handle EmbeddingColumn differently from sequence_numeric_column
            if isinstance(fc, EmbeddingColumn):
                # SequenceFeatures returns a tuple with single item output at position 0
                processed_features[fc_name] = self.seq_features_layers[fc_name](inputs)[0]
            else:
                # For sequence_numeric_column, input is dense float tensor (batch, seq_len, feature_dim)
                dense = inputs[fc_name]
                zero = tf.constant(0, dtype=tf.float32)
                
                # Convert dense tensor to sparse tensor.
                # The reported issue occurs because using dense.shape (static) doesn't have batch dim,
                # so use tf.shape(dense) to get dynamic shape:
                indices = tf.where(tf.not_equal(dense, zero))
                values = tf.gather_nd(dense, indices)
                
                # Use dynamic shape for shape of sparse tensor
                dense_shape = tf.shape(dense)
                sparse = tf.SparseTensor(indices, values, dense_shape)
                
                processed_features[fc_name] = self.seq_features_layers[fc_name]({fc_name: sparse})[0]
        
        # Concatenate all processed features on last dimension
        x = tf.concat(list(processed_features.values()), axis=-1)
        
        x = self.lstm(x)
        x = self.output_layer(x)
        return x

def my_model_function():
    # Construct example feature columns matching the original issue:
    # - One sequence_numeric_column named 'dense'
    # - One sequence_categorical_column_with_identity + embedding named 'categorical', embedding dim=16
    sequence_length = 9
    nb_features = 10
    embedding_units = 16
    
    fc_dense = tf.feature_column.sequence_numeric_column('dense')
    fc_cat = tf.feature_column.sequence_categorical_column_with_identity('categorical', nb_features)
    fc_cat = tf.feature_column.embedding_column(fc_cat, embedding_units)
    
    # We can create a model with both feature columns as an example,
    # but to test specifically the numeric column fix, you can pass [fc_dense] only.
    return MyModel([fc_dense, fc_cat], nb_features=nb_features)

def GetInput():
    # Return an input dictionary matching the expected input for MyModel
    # The inputs have batch dimension B, sequence length 9, feature dims accordingly
    B = 24     # mimic batch_size
    seq_len = 9
    nb_features = 10
    embedding_units = 16  # only needed for construction, not input
    
    # Numeric sequence input 'dense': shape (B, seq_len), float32
    # Use 1-dimensional numeric features per time step
    dense_input = tf.random.uniform((B, seq_len, 1), dtype=tf.float32)
    
    # Categorical sequence input 'categorical': integer IDs in [0, nb_features-1] shape (B, seq_len)
    categorical_input = tf.random.uniform((B, seq_len), minval=0, maxval=nb_features, dtype=tf.int32)
    
    # Compose dictionary
    return {'dense': dense_input, 'categorical': categorical_input}


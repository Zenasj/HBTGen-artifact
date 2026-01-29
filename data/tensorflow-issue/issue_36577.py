# tf.SparseTensor with shape corresponding to a 1D sparse feature input (batch_size, ?) and int64 dtype
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define feature column: categorical with hash bucket, then embedding
        # Here the feature name is 'hisList' and expected as SparseTensor input
        cat_col = tf.feature_column.categorical_column_with_hash_bucket(
            key='hisList', hash_bucket_size=1000, dtype=tf.string)
        emb_col = tf.feature_column.embedding_column(cat_col, dimension=12)
        
        # DenseFeatures layer expects a dictionary input mapping feature keys to tensors
        # Handle sparse=True in input explicitly
        self.feature_layer = tf.keras.layers.DenseFeatures([emb_col])
        
        # Subsequent dense layers
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=None):
        # inputs should be a dictionary with key 'hisList' mapping to a SparseTensor
        x = self.feature_layer(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Build a batch of sparse inputs for 'hisList':
    # Assume batch_size=4, variable length sequences represented as SparseTensor
    # Example sparse data:
    # batch index: [0, 0, 1, 2, 2, 3]
    # values (string): ["11", "102", "33", "1", "44", "123"]
    batch_size = 4
    
    # Sparse indices: [ [batch, index_in_feature], ... ]
    indices = tf.constant([
        [0, 0],
        [0, 1],
        [1, 0],
        [2, 0],
        [2, 1],
        [3, 0]
    ], dtype=tf.int64)
    
    # Sparse values as strings - categorical_column with dtype string
    values = tf.constant(["11", "102", "33", "1", "44", "123"], dtype=tf.string)
    
    # Dense shape for SparseTensor - max batch size and max sequence length (2 here)
    dense_shape = tf.constant([batch_size, 2], dtype=tf.int64)
    
    sparse_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    
    # Return as dictionary keyed by feature name expected by feature columns
    return {'hisList': sparse_input}


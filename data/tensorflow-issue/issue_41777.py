# tf.random.uniform((B, 2), dtype=tf.int64) ← Input features 'm1' and 'm2', each of shape (1,)
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_v2 as fc

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Define feature columns: categorical column with hash bucket for 'm1',
        # embedded into vector of dim 1; numeric column for 'm2'
        m1_cat_col = fc.categorical_column_with_hash_bucket(key='m1', hash_bucket_size=2, dtype=tf.int64)
        m1_emb_col = fc.embedding_column(m1_cat_col, dimension=1, combiner='mean')
        m2_num_col = fc.numeric_column('m2', shape=(1,))
        self.feature_columns = [m1_emb_col, m2_num_col]

        # DenseFeatures layer to process input features into embeddings / tensors
        self.dense_features = tf.keras.layers.DenseFeatures(self.feature_columns, name='d_embedded')

        # Single Dense output layer (binary classification output)
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        """
        inputs: dict with keys 'm1' (int64 tensor, shape (batch,1)),
                              'm2' (int64 tensor, shape (batch,1))
        """
        x = self.dense_features(inputs)  # Extract dense features from input dict
        x = self.output_layer(x)
        return x

def my_model_function():
    # Return an instance of MyModel compiled with Adagrad optimizer and binary crossentropy loss
    dnn_optimizer = tf.keras.optimizers.Adagrad(learning_rate=1)
    model = MyModel()
    model.compile(optimizer=dnn_optimizer, loss='binary_crossentropy', metrics=['binary_crossentropy'])
    return model

def GetInput():
    """
    Generate a batch of input features as a dict of tensors suitable for MyModel.
    - 'm1': int64 tensor with shape (batch_size, 1), categorical input values 0 or 1
    - 'm2': float32 tensor with shape (batch_size, 1), numeric input values
    We'll produce integer values for 'm1' as needed by categorical_column_with_hash_bucket.
    """
    batch_size = 8  # arbitrary batch size

    # 'm1' categorical hash bucket input: random integers 0 or 1, shape (batch_size, 1), dtype int64
    m1_input = tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=2, dtype=tf.int64)

    # 'm2' numeric input: random floats cast to int64 (to match original code dtype), shape (batch_size, 1)
    # NOTE: original code used int64 dtype for numeric input; we keep consistent with that.
    # Casting a float uniform to int64 in [0,100) as example numeric input.
    m2_input_float = tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=100, dtype=tf.float32)
    m2_input = tf.cast(m2_input_float, tf.int64)

    return {'m1': m1_input, 'm2': m2_input}


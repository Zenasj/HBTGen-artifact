# tf.random.uniform((B,)) with dictionary of string/int/float inputs as keys (batch size unknown, each input shape (B,1))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define feature columns
        store_feature = tf.feature_column.categorical_column_with_vocabulary_list(
            'store', vocabulary_list=['a', 'b'])
        store_feature = tf.feature_column.embedding_column(store_feature, dimension=64)

        loc_feature = tf.feature_column.categorical_column_with_vocabulary_list(
            'loc', vocabulary_list=['x', 'y', 'z'])
        loc_feature = tf.feature_column.embedding_column(loc_feature, dimension=32)

        self.feature_columns = [store_feature, loc_feature]

        # Feature layer to convert feature columns to dense tensors
        self.dense_features = tf.keras.layers.DenseFeatures(feature_columns=self.feature_columns)

        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='relu')

    def call(self, inputs):
        """
        inputs is a dict with keys: 'store' and 'loc' containing tensors of shape (B, 1), dtype string
        """
        x = self.dense_features(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    """
    Return a sample input dictionary compatible with MyModel:
    - 'store': tf.string tensor (B,1) with values in ['a', 'b']
    - 'loc': tf.string tensor (B,1) with values in ['x', 'y', 'z']
    Let's pick batch size 4 for example.
    """
    batch_size = 4
    stores = tf.constant([['a'], ['b'], ['a'], ['b']], dtype=tf.string)  # shape (4,1)
    locs = tf.constant([['x'], ['y'], ['z'], ['x']], dtype=tf.string)    # shape (4,1)
    return {'store': stores, 'loc': locs}


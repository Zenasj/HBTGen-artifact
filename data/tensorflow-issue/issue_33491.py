# tf.random.uniform((B,)) with dtype=tf.string and tf.int64 for categorical features input

import tensorflow as tf
from tensorflow import feature_column

# We reconstruct a model reflecting the core issue in the issue:
# The main challenge is handling inputs with different structures:
# - A SparseTensor input corresponding to a categorical string feature ('h_k_u_watchanch_his')
# - A dense int64 input for 'a_gender'
# We use tf.keras.layers.DenseFeatures with an embedding_column from a categorical_column_with_hash_bucket.
# The output is a sigmoid binary classifier.

# The issue described involves mismatched nested structures when feeding input to the model.
# To fix, inputs should be passed as a dict mapping input names to tensors matching input signature shapes.
# Sparse inputs should be represented as tf.SparseTensor and dense inputs as dense Tensors.
# During model construction, specifying input signatures that handle sparse inputs properly is needed.

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define feature column for categorical sparse input, hashed
        self.thal = feature_column.categorical_column_with_hash_bucket(
            'h_k_u_watchanch_his', hash_bucket_size=100)
        # Use embedding_column with dimension 10 and combiner mean
        self.thal_one_hot = feature_column.embedding_column(
            self.thal, dimension=10, combiner='mean')
        self.feature_columns = [self.thal_one_hot]

        # Create DenseFeatures layer from feature columns
        self.dense_feature_layer = tf.keras.layers.DenseFeatures(self.feature_columns, name='DenseFeatures')

        # Output dense layer
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        # inputs is a dict: keys are feature column names
        # 'h_k_u_watchanch_his' : SparseTensor of strings
        # 'a_gender': dense int64 Tensor (not used in feature columns here, but let's keep consistent input dict)
        
        # DenseFeatures layer expects a dict with the input features matching the feature_column keys
        # Our feature_columns only use 'h_k_u_watchanch_his'
        features_out = self.dense_feature_layer(inputs)
        output = self.output_layer(features_out)
        return output


def my_model_function():
    # Return an instance of MyModel with compiled optimizer, loss, metrics, ready for training
    model = MyModel()
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def GetInput():
    # Return a valid input dict with proper SparseTensor and Tensor matching the inputs expected by MyModel

    # Batch size is arbitrary, here using batch_size=2 for example clarity

    # Simulate 'h_k_u_watchanch_his' SparseTensor input representing variable-length string features:
    # SparseTensor consists of three components: indices, values, dense_shape
    # Let's create a batch of 2 examples:
    # Example 1 has 2 strings, Example 2 has 1 string
    indices = tf.constant([[0, 0], [0, 1], [1, 0]], dtype=tf.int64)
    values = tf.constant(['foo', 'bar', 'baz'], dtype=tf.string)
    dense_shape = tf.constant([2, 2], dtype=tf.int64)

    sparse_input = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    # 'a_gender' is a dense int64 tensor with shape (batch_size, 1)
    a_gender = tf.constant([[1], [0]], dtype=tf.int64)

    # The inputs dict matches the input layer names used for DenseFeatures and model inputs
    inputs = {
        'h_k_u_watchanch_his': sparse_input,
        'a_gender': a_gender
    }
    return inputs


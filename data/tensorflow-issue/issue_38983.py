# tf.random.uniform((B,), dtype=tf.int32) ← input is a dict of 1D integer tensors (sparse indices) keyed by feature names

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Add, Input
from tensorflow.keras import Model

class MyModel(tf.keras.Model):
    def __init__(self, list_features=None):
        super().__init__()
        # Default feature list if none provided
        if list_features is None:
            list_features = ['feature_1', 'feature_2']
        self.list_features = list_features
        # Create an embedding layer per feature
        self.embeddings = {feat: Embedding(10, 3, name=f"embedding_{feat}") for feat in list_features}
        self.add_layer = Add()

    def call(self, inputs):
        """
        inputs: dict of feature_name -> tf.Tensor (int32 indices)
        
        The call enforces that all expected features in self.list_features
        are present in inputs, otherwise raises an assertion error.
        
        It ignores extra keys in the input dict that are not in list_features.
        
        Output:
          Summation of embeddings for all features.
        """
        # Verify inputs include exactly the required keys (allowing extra keys, but required keys must exist)
        for feat in self.list_features:
            # This check enforces the expected behavior: fail immediately if a required feature key is missing.
            tf.debugging.assert_equal(
                tf.reduce_any([tf.equal(feat, key) for key in inputs.keys()]),
                True,
                message=f"Input is missing required feature '{feat}'"
            )
        # Compute embeddings for features in defined order to ensure consistent output despite input dict order
        tensors = [self.embeddings[feat](inputs[feat]) for feat in self.list_features]
        return self.add_layer(tensors)

def my_model_function():
    # Instantiate MyModel with default features "feature_1" and "feature_2"
    return MyModel()

def GetInput():
    # Returns a dict matching the expected input for MyModel: two features with integer ids
    # Using batch size 3 as example
    batch_size = 3
    feature_1 = tf.random.uniform((batch_size,), minval=0, maxval=9, dtype=tf.int32)
    feature_2 = tf.random.uniform((batch_size,), minval=0, maxval=9, dtype=tf.int32)
    return {
        'feature_1': feature_1,
        'feature_2': feature_2
    }


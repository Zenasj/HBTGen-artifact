# tf.random.uniform((B, 1), dtype=tf.string) ← Input is a batch of string features named "sex" with shape (batch_size, 1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a categorical_column_with_vocabulary_file feature column
        # related to the issue: the vocabulary file should be included as an asset
        # to ensure the saved model does not depend on the original file path.
        #
        # Note: The issue highlights that when num_oov_buckets=0, the asset is not saved.
        # So here, we set num_oov_buckets=5 as a practical workaround to force asset export.
        #
        # Vocabulary file path is assumed relative or absolute. For demonstration, 
        # we set the path as "./sex.txt" (user should prepare this file in saving environment).
        self.vocab_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key="sex",
            vocabulary_file="./sex.txt",
            num_oov_buckets=5  # Ensure asset is saved/exported
        )
        # Embed categorical column to size 2 vectors
        self.embedding_column = tf.feature_column.embedding_column(self.vocab_column, dimension=2)
        # DenseFeatures layer to transform input dictionary of features based on feature columns
        self.dense_features = tf.keras.layers.DenseFeatures([self.embedding_column])

    def call(self, inputs, training=False):
        # inputs expected as a dictionary from feature name to tensor,
        # e.g. {"sex": tf.Tensor}
        return self.dense_features(inputs)

def my_model_function():
    # Return an initialized instance of MyModel.
    # The model expects the vocab file './sex.txt' to be present in the runtime environment.
    # When saving this model with tf.saved_model.save, the asset file ./sex.txt should be
    # collected automatically as part of the SavedModel assets due to num_oov_buckets > 0.
    return MyModel()

def GetInput():
    # Create dummy input tensor matching the expected input:
    # Batch size 4, single string feature per example.
    # Random strings chosen from a small set consistent with expected vocab.
    sample_values = tf.constant([["male"], ["female"], ["other"], ["unknown"]], dtype=tf.string)
    # Returned as a dict with key 'sex' to match the model's preprocessing input.
    return {"sex": sample_values}


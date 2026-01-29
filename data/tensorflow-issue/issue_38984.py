# tf.ragged.constant([[...]], shape=[None]) for each feature in dict input (ragged=True, shape=[None])

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Add, Lambda, Input
from tensorflow.keras.models import Model

class MyModel(tf.keras.Model):
    def __init__(self, list_features=None, **kwargs):
        super().__init__(**kwargs)
        # The model merges embeddings from a dict of ragged inputs of shape [batch_size, None] per feature
        if list_features is None:
            list_features = ['feature_1', 'feature_2']

        self.list_features = list_features
        # Initialize embeddings for each feature: 10 tokens, embedding dim 3
        self.embeddings = {feature: Embedding(10, 3) for feature in list_features}
        # Lambda layer to compute mean over the sequence dimension axis=1
        self.mean = Lambda(lambda x: tf.reduce_mean(x, axis=1))

        # Important: signal that layer supports ragged inputs
        self._supports_ragged_inputs = True

    def call(self, inputs, training=False):
        # inputs is a dictionary of ragged tensors keyed by feature name
        # Apply embedding to each ragged input tensor
        embedded_tensors = []
        for feature in self.list_features:
            x = inputs[feature]  # Expected shape: RaggedTensor with ragged dims on axis=1
            emb = self.embeddings[feature](x)  # shape: (batch_size, None, embedding_dim)
            mean_emb = self.mean(emb)  # shape: (batch_size, embedding_dim)
            embedded_tensors.append(mean_emb)
        # Sum all mean embeddings elementwise across features
        output = Add()(embedded_tensors)
        return output

def my_model_function():
    # Return an instance of MyModel with default features
    return MyModel()

def GetInput():
    # Create a dict of ragged tensors matching input to MyModel:
    # Each feature is a RaggedTensor with shape [batch_size, None], dtype int32 (token indices)
    # Use batch_size=2, variable-length sequences per batch element as example
    feature_1 = tf.ragged.constant([[0], [1, 3]], dtype=tf.int32)
    feature_2 = tf.ragged.constant([[1, 2], [4]], dtype=tf.int32)
    inputs = {'feature_1': feature_1, 'feature_2': feature_2}
    return inputs


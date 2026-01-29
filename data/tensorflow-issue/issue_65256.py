# tf.random.uniform((1,), dtype=tf.string), tf.random.uniform((1,), dtype=tf.string), ... ← Input dict keys each with shape (1,), dtype=tf.string

import tensorflow as tf
import pandas as pd
from typing import Dict, List, Tuple, Text, Union

class MyModel(tf.keras.Model):
    """
    Fusion of a two-tower recommendation brute-force retrieval layer built around 
    user query_model and candidate_model embeddings, with candidate scoring and top-k retrieval.
    
    This is reconstructed from the issue about TensorFlow saved model input signatures and infinite loops 
    when trying to save with tf.function decorator.
    
    Assumptions/Notes:
    - The input 'queries' is a dict of string tensors each with shape (1,) as per positions_inputs.
    - The 'candidates_raw' is a list of dicts that can be converted to a pandas DataFrame.
    - The model has .query_model and .candidate_model attributes for generating embeddings.
    - The forward pass encodes candidates, generates embeddings, computes scores with matmul,
      and emits top-k scores and candidate identifiers.
    - We do batching and dataset conversions within call to handle candidate embedding efficiently.
    """

    def __init__(self, model, k=5, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.k = k
        self._k = k
        
        # Placeholders for candidate embeddings and identifiers to be stored as weights for exporting
        self._candidates = None
        self._identifiers = None
        
        # positions_inputs signature reconstruct based on keys given in chunk1
        self.positions_inputs = {
            'key_1': tf.TensorSpec(shape=(1,), dtype=tf.string),
            'key_2': tf.TensorSpec(shape=(1,), dtype=tf.string),
            'key_3': tf.TensorSpec(shape=(1,), dtype=tf.string),
            'key_4': tf.TensorSpec(shape=(1,), dtype=tf.string),
            'key_5': tf.TensorSpec(shape=(1,), dtype=tf.string),
            'key_6': tf.TensorSpec(shape=(1,), dtype=tf.string),
            'key_7': tf.TensorSpec(shape=(1,), dtype=tf.string),
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)] * 7 + [tf.TensorSpec(shape=(), dtype=tf.int32)])
    def call(
        self, 
        key_1: tf.Tensor,
        key_2: tf.Tensor,
        key_3: tf.Tensor,
        key_4: tf.Tensor,
        key_5: tf.Tensor,
        key_6: tf.Tensor,
        key_7: tf.Tensor,
        k: tf.Tensor = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass that:
        - Accepts 7 input string tensors each shape (None,) - batched queries keys
        - Collects all inputs into a dict query
        - Uses self.model.query_model to compute query embeddings
        - Builds candidate embeddings from self._candidates weight set during index building
        - Computes scores by matmul of query and candidate embeddings
        - Returns top-k scores and associated candidate identifiers
        
        Note: The original code expects candidates passed in call as well, but for signature simplicity, here 
        we assume candidates are pre-indexed via an explicit index call or as weights.
        """
        queries = {
            'key_1': key_1,
            'key_2': key_2,
            'key_3': key_3,
            'key_4': key_4,
            'key_5': key_5,
            'key_6': key_6,
            'key_7': key_7,
        }

        # Stack string tensors to shape (batch, 1) as in original expected input possibly
        queries = {k: tf.reshape(v, [-1, 1]) for k, v in queries.items()}

        # Concatenate or process queries into embedding input as the original user model expects
        # The original model likely does concatenation and normalization of embeddings - we call it directly
        # Here we simulate it by creating a dummy tensor from query strings via casting because underlying model unknown
        # But since model.query_model expects tensor input, we assume it is a preprocessing/Keras model handling that internally
        
        # Convert dict input to some representation for query_model - assuming it handles dict of string tensors
        # We emit a dummy combined tensor by concatenation of string tensors casted to int for demonstration
        # In real use case, the original `self.model.query_model` will likely have input layers accepting dict of string tensors.
        # So here just forward the dict and rely on user's model to handle it.

        query_embeddings = self.model.query_model(queries)

        # Defensive checks for candidate embeddings weight
        if self._candidates is None or self._identifiers is None:
            raise ValueError("Candidates index not built. You must call `index()` first to create retrieval index.")

        # Number of candidates and queries
        candidate_count = tf.shape(self._candidates)[0]
        query_count = tf.shape(query_embeddings)[0]

        # Compute scores between queries and candidates using matmul
        scores = tf.matmul(query_embeddings, self._candidates, transpose_b=True)  # shape (query_count, candidate_count)

        k = k if k is not None else self._k

        # Pick top-k per query
        values, indices = tf.math.top_k(scores, k=k)

        # Gather candidate identifiers per top indices
        top_identifiers = tf.gather(self._identifiers, indices)

        return values, top_identifiers

    def index(self, candidates_raw: List[Dict[Text, List[str]]]):
        """
        Index the candidates:
        - Accepts a list of dicts of candidate features with string lists
        - Converts to tf.data.Dataset
        - Maps candidate_model over the dataset to get embeddings
        - Stores embeddings and identifiers as weights for serving

        This method must be called before serving and calling model.
        """
        # Convert candidates_raw list of dicts [ {feature: [val1, val2]} ] to pandas DataFrame
        df = pd.DataFrame.from_records(candidates_raw)
        
        # Convert dataframe columns to list of feature values (since candidate_model expects batched input)
        # pandas `from_records` with list of dicts yields columns mapping to list of feature values
        
        # Create dataset from dict of lists with shape [num_candidates]
        ds = tf.data.Dataset.from_tensor_slices(dict(df))
        
        # Batch candidates
        batch_size = 32  # reasonable batching
        ds = ds.batch(batch_size)
        
        # Map candidate_model to get embeddings
        embedded_ds = ds.map(self.model.candidate_model)
        
        # We assume identifiers as the range index for candidates
        identifiers = tf.range(df.shape[0], dtype=tf.int32)

        # Collect embeddings into one tensor
        candidate_embeddings = tf.concat(list(embedded_ds), axis=0)
        
        # Save embeddings and identifiers as weights on the model for later serving
        # Create or assign these weights only once or upon reindexing
        if self._candidates is None:
            self._candidates = self.add_weight(
                name="candidates",
                shape=candidate_embeddings.shape,
                dtype=candidate_embeddings.dtype,
                initializer=tf.keras.initializers.Zeros(),
                trainable=False,
            )
        if self._identifiers is None:
            self._identifiers = self.add_weight(
                name="identifiers",
                shape=identifiers.shape,
                dtype=identifiers.dtype,
                initializer=tf.keras.initializers.Zeros(),
                trainable=False,
            )

        self._candidates.assign(candidate_embeddings)
        self._identifiers.assign(identifiers)
        

def my_model_function():
    """
    Returns an instance of MyModel.
    Assumes the user provides a 'model' with .query_model and .candidate_model attributes.
    Here we prepare a dummy stub model as placeholder to make the class complete.
    """
    # Define dummy query_model and candidate_model for demonstration

    class DummyQueryModel(tf.keras.Model):
        def call(self, inputs):
            # Inputs is dict of (batch,1) string tensors, just embed by string length for demo
            # Convert string to byte length and cast to float embedding vector for dummy behavior
            # This is placeholder logic only
            lengths = []
            for k in sorted(inputs.keys()):
                s = inputs[k]  # shape (batch,1), dtype string
                l = tf.strings.length(s)
                lengths.append(tf.cast(l, tf.float32))
            # Concatenate all length features => shape (batch, features)
            return tf.stack(lengths, axis=1)

    class DummyCandidateModel(tf.keras.Model):
        def call(self, inputs):
            # inputs is dict of feature string tensors with batch dimension
            # same dummy embed as above: length of each string feature concatenated
            lengths = []
            for k in sorted(inputs.keys()):
                s = inputs[k]  # shape (batch,)
                l = tf.strings.length(s)
                lengths.append(tf.cast(l, tf.float32))
            return tf.stack(lengths, axis=1)

    class DummyModel:
        def __init__(self):
            self.query_model = DummyQueryModel()
            self.candidate_model = DummyCandidateModel()

    model = DummyModel()
    return MyModel(model=model, k=5)


def GetInput():
    """
    Generate input compatible with MyModel.call input_signature:
    Seven string tensors each shape (batch,) = (1,)
    and one int32 scalar for k.
    """
    batch_size = 1
    key_1 = tf.constant(["example1"], dtype=tf.string)
    key_2 = tf.constant(["example2"], dtype=tf.string)
    key_3 = tf.constant(["example3"], dtype=tf.string)
    key_4 = tf.constant(["example4"], dtype=tf.string)
    key_5 = tf.constant(["example5"], dtype=tf.string)
    key_6 = tf.constant(["example6"], dtype=tf.string)
    key_7 = tf.constant(["example7"], dtype=tf.string)
    k = tf.constant(2, dtype=tf.int32)
    return key_1, key_2, key_3, key_4, key_5, key_6, key_7, k


import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

@tf.keras.saving.register_keras_serializable(package="MyLayers")
class BruteForce2(tf.keras.layers.Layer):
  """Brute force retrieval."""

  def __init__(self, model, k=5, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.k = k
        self._k = k
        self._candidates = None
        
  def _compute_score(self, queries: tf.Tensor,
                     candidates: tf.Tensor) -> tf.Tensor:

        print(queries, candidates)
    
        return tf.matmul(queries, candidates, transpose_b=True)
  
  def _topk_ds(self, data):
    df = data.copy()
    df = {key: np.array(value)[:,tf.newaxis] for key, value in df.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df)))
    ds.prefetch(data.size)

    return ds
  
  positions_inputs = {
                      'key_1': tf.TensorSpec(shape=(1,), dtype=tf.string), 
                      'key_2': tf.TensorSpec(shape=(1,), dtype=tf.string), 
                      'key_3': tf.TensorSpec(shape=(1,), dtype=tf.string), 
                      'key_4': tf.TensorSpec(shape=(1,), dtype=tf.string), 
                      'key_5': tf.TensorSpec(shape=(1,), dtype=tf.string), 
                      'key_6': tf.TensorSpec(shape=(1,), dtype=tf.string), 
                      'key_7': tf.TensorSpec(shape=(1,), dtype=tf.string)
                      }
  
  @tf.function(input_signature=[positions_inputs, [candidaster.element_spec], tf.TensorSpec(shape=(), dtype=tf.int32)])
  def _encode(self, queries, raw_candidates, k):
    # data = pd.DataFrame(raw_candidates)
    # data.head()
    # returnment = self._topk_ds(data)
    return queries, raw_candidates, k
  
  # @tf.function(input_signature=[
  #     positions_inputs, 
  #     tf.TensorSpec(shape=[None], dtype=tf.int64),
  #     tf.TensorSpec(shape=(), dtype=tf.int64)
  #   ])
  @tf.function(input_signature=[positions_inputs, [candidaster.element_spec], tf.TensorSpec(shape=(), dtype=tf.int32)])
  def call(self, queries: Union[tf.Tensor, Dict[Text, tf.Tensor]], candidates_raw: List[tf.Tensor], k: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    
    # queries, candidates_raw, k = self._encode(queries, candidates_raw, k)
    # candidates_raw = self._topk_ds(pd.DataFrame(candidates_raw))
    # candidates_raw = self._encode(candidates_raw)
    
    parse_one = pd.DataFrame.from_dict(candidates_raw).to_dict(orient='list')
    candidates_raw = tf.data.Dataset.from_tensor_slices(parse_one)
    
    candidates = tf.data.Dataset.zip(candidates_raw.batch(1).map(lambda x: x['experience.jobTitle']), 
                                           candidates_raw.batch(1).map(self.model.candidate_model)
                                          )
    
    spec = candidates.element_spec

    if isinstance(spec, tuple):
      identifiers_and_candidates = list(candidates)
      candidates = tf.concat(
          [embeddings for _, embeddings in identifiers_and_candidates],
          axis=0
      )
      identifiers = tf.concat(
          [identifiers for identifiers, _ in identifiers_and_candidates],
          axis=0
      )
    else:
      candidates = tf.concat(list(candidates), axis=0)
      identifiers = None
    
    # self._candidates = candidates

    if identifiers is None:
      identifiers = tf.range(candidates.shape[0])
    if tf.rank(candidates) != 2:
      raise ValueError(
          f"The candidates tensor must be 2D (got {candidates.shape}).")
    if candidates.shape[0] != identifiers.shape[0]:
      raise ValueError(
          "The candidates and identifiers tensors must have the same number of rows "
          f"(got {candidates.shape[0]} candidates rows and {identifiers.shape[0]} "
          "identifier rows). "
      )
    # We need any value that has the correct dtype.
    identifiers_initial_value = tf.zeros((), dtype=identifiers.dtype)
    self._identifiers = self.add_weight(
        name="identifiers",
        dtype=identifiers.dtype,
        shape=identifiers.shape,
        initializer=tf.keras.initializers.Constant(
            value=identifiers_initial_value),
        trainable=False)
    self._candidates = self.add_weight(
        name="candidates",
        dtype=candidates.dtype,
        shape=candidates.shape,
        initializer=tf.keras.initializers.Zeros(),
        trainable=False)
    self._identifiers.assign(identifiers)
    self._candidates.assign(candidates)
    # self._reset_tf_function_cache()
    

    k = k if k is not None else self._k

    if self._candidates is None:
      raise ValueError("The `index` method must be called first to "
                       "create the retrieval index.")

    if self.model.query_model is not None:
      queries = self.model.query_model(queries)

    scores = self._compute_score(queries, self._candidates)

    values, indices = tf.math.top_k(scores, k=k)

    return values, tf.gather(self._identifiers, indices)

# save model
custom_index = BruteForce2(final_model, k=5)

for l in data_ds.take(1):
    # scores, chose = custom_index(l, mapped_candidates, 2)

    # custom_index.call = tf.function(custom_index.call, input_signature=[positions_inputs, [candidaster.element_spec], tf.TensorSpec(shape=(), dtype=tf.int32)])
    # print(scores, chose)
    
    custom_index.model.task_retrieval = tfrs.tasks.Retrieval()
    
    signatures = {"serving_default": custom_index.call.get_concrete_function()}
    
    tf.saved_model.save(custom_index, "./serveModels/exportedMode19", signatures=signatures)
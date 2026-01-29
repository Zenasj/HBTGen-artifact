# tf.random.uniform((B=32, T=32, C=5), dtype=tf.float32) ← inferred input shape from the example data

import tensorflow as tf

FEATURES_DIM = 5

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # SequenceFeatures expects a list and returns a list of sequence tensors
        self.features = tf.keras.experimental.SequenceFeatures([
            tf.feature_column.sequence_numeric_column('features', shape=(FEATURES_DIM,))
        ])
        # Two TimeDistributed Dense layers to process the sequence features
        self.dense_1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(32, activation='relu'))
        self.dense_2 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(1))
        
        # Add explicit input specification for sparse input to enable shape inference
        # This is needed because subclassed models must have explicit tf.Input layers
        self._input_spec = tf.keras.Input(
            shape=(None, FEATURES_DIM),  # Temporal dimension is variable length
            name="features",
            sparse=True,
            dtype=tf.float32)
        # _add_inputs is not exposed in TF 2.0; this workaround sets input_spec manually
        self._input_spec = self._input_spec

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # inputs expected as a dict: {'features': SparseTensor}
        # SequenceFeatures produces a list with processed dense Tensor
        outputs = self.features(inputs)
        # outputs is a list; we take the first element (processed dense sequence tensor)
        x = outputs[0]
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x

    def compute_output_shape(self, input_shape):
        # input_shape expected as (batch_size, time_steps, FEATURES_DIM)
        shape = tf.TensorShape(input_shape).as_list()
        # TimeDistributed layer produces output with last dimension = 1
        shape[-1] = 1
        return tf.TensorShape(shape)


def my_model_function():
    return MyModel()

def GetInput():
    # Create a random dense Tensor of shape (batch=32, time=32, features=5)
    # SparseTensor input needs to use tf.sparse.stype for sparse input demonstration
    import numpy as np
    
    batch_size = 32
    time_steps = 32
    features = FEATURES_DIM
    
    # Create a random dense input
    dense_input = tf.random.uniform((batch_size, time_steps, features), dtype=tf.float32)
    
    # Convert to SparseTensor by zeroing out some values arbitrarily
    mask = tf.math.greater(dense_input, 0.5)
    indices = tf.where(mask)
    values = tf.gather_nd(dense_input, indices)
    sparse_shape = tf.constant([batch_size, time_steps, features], dtype=tf.int64)
    sparse_input = tf.SparseTensor(indices=indices, values=values, dense_shape=sparse_shape)
    
    # Model expects a dict input keyed by 'features'
    return {'features': sparse_input}


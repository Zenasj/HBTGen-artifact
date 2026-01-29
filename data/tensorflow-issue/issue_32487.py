# tf.random.uniform((batch_size, 1), dtype=tf.float32) ← input shape inferred from features: dense feature shape (batch_size, 1), sparse inputs sparse tensor of shape (batch_size, 1)

import tensorflow as tf
import numpy as np

class SparseSlice(tf.keras.layers.Layer):
    def __init__(self, feature_column):
        super(SparseSlice, self).__init__()
        self.fc = feature_column

    def build(self, input_shape):
        # Kernel shape: (num_buckets,)
        self.kernel = self.add_weight(
            '{}_kernel'.format(self.fc.name),
            shape=(self.fc.num_buckets,),
            dtype=tf.float32,
            initializer='random_uniform')

    def call(self, input):
        # Use the feature_column method to transform input SparseTensor to ids
        ids = self.fc._transform_input_tensor(input)
        # Gather embedding vector for each id, expand dims so shape (batch_size, 1, 1)
        return tf.expand_dims(tf.gather(self.kernel, ids.values), axis=1)

class MyModel(tf.keras.Model):
    def __init__(self, batch_size=10):
        super(MyModel, self).__init__()
        self.batch_size = batch_size

        # Define feature columns
        self.sparse_col = tf.feature_column.categorical_column_with_hash_bucket(
            'sparse_col', 10000, dtype=tf.int64)
        self.dense_col = tf.feature_column.numeric_column(
            'dense_col', dtype=tf.float32)
        
        # SparseSlice custom layer
        self.sparse_slice = SparseSlice(self.sparse_col)
        
        # Dense layer on sparse embed output
        self.dense1 = tf.keras.layers.Dense(1, activation='sigmoid')
        
        # Dense features layer for dense inputs
        self.dense_features = tf.keras.layers.DenseFeatures([self.dense_col])
        
        # Concatenate layer
        self.concat = tf.keras.layers.Concatenate()
        
        # Final Dense layer with sigmoid activation
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')

        # Inputs layers compatible with the original example:
        # sparse_inputs is SparseTensor input of shape (batch_size, None)
        # dense_inputs is dict containing dense_col input of shape (batch_size, 1)
        self.sparse_inputs = tf.keras.layers.Input(
            name=self.sparse_col.name,
            shape=(None,),  # variable length in sparse dimension
            batch_size=self.batch_size,
            sparse=True,
            dtype=tf.int64)
        self.dense_inputs = {
            self.dense_col.name: tf.keras.layers.Input(
                name=self.dense_col.name,
                shape=(1,),
                dtype=tf.float32)
        }
        # Build model call function graph once for proper shape inference
        self._build_model_graph()

    def _build_model_graph(self):
        # Build the layers with inputs for shape inference
        sparse_out = self.sparse_slice(self.sparse_inputs)
        x1 = self.dense1(sparse_out)
        x2 = self.dense_features(self.dense_inputs)
        concatenated = self.concat([x1, x2])
        output = self.dense2(concatenated)
        # Create tf.keras.Model for clarity if needed internally
        self.keras_model = tf.keras.Model(
            inputs=[self.dense_inputs, { 'sparse_output': self.sparse_inputs }],
            outputs=output)

    def call(self, inputs):
        # Inputs is tuple/dict with keys: dense_col -> dense tensor, sparse_col -> SparseTensor
        # Unpack inputs
        # Accept inputs either as dict with keys matching feature columns or as tuple
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            dense_inputs = inputs[0]
            sparse_inputs = inputs[1]['sparse_output']
        elif isinstance(inputs, dict):
            # Because in original usage sparse_inputs passed as dict with key sparse_output
            if 'sparse_output' in inputs:
                sparse_inputs = inputs['sparse_output']
                # The dense inputs dict should have dense_col key
                dense_inputs = {k: v for k, v in inputs.items() if k != 'sparse_output'}
            else:
                # fallback: assume keys directly map
                sparse_inputs = inputs.get(self.sparse_col.name)
                dense_inputs = {self.dense_col.name: inputs.get(self.dense_col.name)}
        else:
            raise ValueError("Unsupported input format")

        sparse_out = self.sparse_slice(sparse_inputs)
        x1 = self.dense1(sparse_out)
        x2 = self.dense_features(dense_inputs)
        concatenated = self.concat([x1, x2])
        output = self.dense2(concatenated)
        return output


def my_model_function():
    # Return an instance of MyModel initialized with default batch_size=10
    model = MyModel()
    # Compile with same configuration as in issue
    model.compile(optimizer='adam', loss='mse')
    return model

def GetInput():
    batch_size = 10
    # Sparse inputs: SparseTensor with shape (batch_size, 1)
    indices = [[i, 0] for i in range(batch_size)]
    values = np.random.randint(0, 1000, size=(batch_size,)).astype(np.int64)
    dense_shape = (batch_size, 1)
    sparse_tensor = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    # Dense inputs: dict with key dense_col and shape (batch_size, 1)
    dense_values = tf.constant(np.random.random((batch_size, 1)), dtype=tf.float32)

    # Return inputs in the format matching model call, which expects a tuple:
    # (dense_inputs_dict, {'sparse_output': sparse_inputs})
    return (
        { 'dense_col': dense_values },
        { 'sparse_output': sparse_tensor }
    )


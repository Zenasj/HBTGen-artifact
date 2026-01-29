# tf.sparse.SparseTensor with dense shape (B, 40, 1), dtype=tf.float32

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # SequenceFeatures expects a list of feature_columns describing sparse sequence inputs
        self.features = [
            tf.feature_column.sequence_numeric_column('f1', dtype=tf.float32),
            tf.feature_column.sequence_numeric_column('f2', dtype=tf.float32)
        ]
        self.seq_feature_layer = tf.keras.experimental.SequenceFeatures(self.features)
        self.lstm = tf.keras.layers.LSTM(128, return_sequences=False)
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense = tf.keras.layers.Dense(1, activation='tanh')

    def call(self, inputs, training=False):
        # The inputs dict expects SparseTensor entries 'f1' and 'f2'
        x, _ = self.seq_feature_layer(inputs)
        x = self.lstm(x)
        x = self.dropout(x, training=training)
        x = self.dense(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    batch = 4  # example batch size, must match training batch size
    # Create random dense numpy arrays, then convert to SparseTensors, simulating sparse sequence inputs
    dense_f1 = np.random.random((batch, 40, 1)).astype(np.float32)
    dense_f2 = np.random.random((batch, 40, 1)).astype(np.float32)
    
    # Convert dense to SparseTensor representations
    sparse_f1 = tf.sparse.from_dense(dense_f1)
    sparse_f2 = tf.sparse.from_dense(dense_f2)
    
    # Return a dict of SparseTensors matching model input signature
    return {'f1': sparse_f1, 'f2': sparse_f2}


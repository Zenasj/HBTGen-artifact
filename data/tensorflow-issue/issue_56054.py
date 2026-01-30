from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
class TestModel(tf.keras.models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sl = tf.keras.layers.StringLookup()
        
    def adapt(self, df):
        self.sl.adapt(df)
    
        
    def call(self, x):
        return self.sl(x)

inp = tf.repeat(tf.constant(['A', 'B', 'C']), 10)
df_inp = tf.data.Dataset.from_tensor_slices(inp)

test_model  = TestModel()
test_model.adapt(df_inp)
print(test_model.get_weights())

test_model.save_weights('tmp/check_weights')

test_model_recon = TestModel()
test_model_recon.load_weights('tmp/check_weights')
print(test_model_recon.get_weights())

import tensorflow as tf
StringLookup = tf.keras.layers.StringLookup

def _uninitialized_lookup_table(self):
  with tf.init_scope():
    return tf.lookup.experimental.MutableHashTable(
        key_dtype=self._key_dtype, value_dtype=self._value_dtype, default_value=self._default_value
      )
def _lookup_table_from_tokens(self, tokens):
  with tf.init_scope():
    lookup_table = self._uninitialized_lookup_table()
    token_start = self._token_start_index()
    token_end = token_start + tf.size(tokens)
    indices = tf.range(token_start, token_end, dtype=tf.int64)
    keys, values = (indices, tokens) if self.invert else (tokens, indices)
    lookup_table.insert(keys, values)
    return lookup_table
StringLookup._uninitialized_lookup_table = _uninitialized_lookup_table
StringLookup._lookup_table_from_tokens = _lookup_table_from_tokens

class TestModel(tf.keras.models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sl = StringLookup()
        
    def adapt(self, df):
        self.sl.adapt(df)
    
        
    def call(self, x):
        return self.sl(x)

inp = tf.repeat(tf.constant(['A', 'B', 'C']), 10)
df_inp = tf.data.Dataset.from_tensor_slices(inp)

test_model  = TestModel()
test_model.adapt(df_inp)
print(test_model.get_weights())
test_model.save_weights('tmp/check_weights')
test_model_recon = TestModel()
test_model_recon.load_weights('tmp/check_weights')
print(test_model_recon.get_weights())
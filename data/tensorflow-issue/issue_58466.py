import numpy as np
import tensorflow as tf

from tensorflow import keras as K


class HashtableEmb(K.Model):
    def __init__(self, training=True, key_dtype=tf.int64, value_dtype=tf.int32):
        K.Model.__init__(self)
        self.training = training
        self.key_dtype = key_dtype
        self.value_dtype = value_dtype

    def build(self, input_shape):
        self.DEFAULT_KEY = 0
        self.DEFAULT_VALUE = 0
        self.default_key = tf.constant(self.DEFAULT_KEY, dtype=self.key_dtype)
        self.empty_key = np.iinfo(np.int64).max
        self.deleted_key = np.iinfo(np.int64).min
        self.default_value = tf.constant(self.DEFAULT_VALUE, dtype=self.value_dtype)
        kw = dict(key_dtype=self.key_dtype, value_dtype=self.value_dtype,
                  default_value=self.default_value, empty_key=self.empty_key, deleted_key=self.deleted_key)
        self.index_table = tf.lookup.experimental.DenseHashTable(**kw)
        if self.training:
            self.index_table.insert(self.default_key, self.default_value)
        self.default_ids_queue = tf.queue.FIFOQueue(2_000_000, self.key_dtype, shapes=[],
                                                    name=f'default_ids_queue')
        self.built = True

    def call(self, ids, training=None):
        flatten_ids = tf.reshape(ids, [-1])
        ids_value = self.index_table.lookup(flatten_ids)
        if training: self.update_stat(flatten_ids)
        ids_index_orig = tf.reshape(ids_value, tf.shape(ids))
        return ids_index_orig

    def get_config(self):
        return dict(
            training=self.training,
            key_dtype=self.key_dtype,
            value_dtype=self.value_dtype,
            **(super(K.Model, self).get_config()))

    def update_stat(self, flatten_ids):
        flatten_ids = tf.identity(flatten_ids)
        self.default_ids_queue.enqueue_many(flatten_ids)  # comment this line, no exception


class TestModel(tf.keras.Model):
    def __init__(self):
        super(TestModel, self).__init__()

    def build(self, input_shape):
        self.hashemb = HashtableEmb()
        self.built = True

    def call(self, x, training=None):
        return {'y_click': self.hashemb(x['a'], training=training)}
 

if __name__ == '__main__':
    m = TestModel()
    print(m({'a': tf.convert_to_tensor([[2], [200]], dtype=tf.int64)}))
    signatures = {'serving_default': tf.function(m.call).get_concrete_function(
        x={'a': tf.TensorSpec([None, 1], dtype=tf.int64)}, training=False)}
    tf.saved_model.save(m, r'G:\t\test_model_save', signatures=signatures)
    m2 = tf.saved_model.load(r'G:\t\test_model_save')
    print(m2({'a': tf.convert_to_tensor([[2], [200]], dtype=tf.int64)}))
# tf.random.uniform((B, 1), dtype=tf.int64) ← Input is a dict with key 'a' holding a 2D int64 tensor of shape (batch_size, 1)

import numpy as np
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, training=True, key_dtype=tf.int64, value_dtype=tf.int32):
        super().__init__()
        self.training = training
        self.key_dtype = key_dtype
        self.value_dtype = value_dtype

        # Constants used for DenseHashTable
        self.DEFAULT_KEY = 0
        self.DEFAULT_VALUE = 0
        self.empty_key = np.iinfo(np.int64).max
        self.deleted_key = np.iinfo(np.int64).min

    def build(self, input_shape):
        # Setup constants as tf constants to be safe for tracing and save/load
        self.default_key = tf.constant(self.DEFAULT_KEY, dtype=self.key_dtype)
        self.default_value = tf.constant(self.DEFAULT_VALUE, dtype=self.value_dtype)

        # Initialize DenseHashTable from tf.lookup.experimental
        self.index_table = tf.lookup.experimental.DenseHashTable(
            key_dtype=self.key_dtype,
            value_dtype=self.value_dtype,
            empty_key=self.empty_key,
            deleted_key=self.deleted_key,
            default_value=self.default_value
        )
        if self.training:
            # Insert the default key-value pair for training
            self.index_table.insert(self.default_key, self.default_value)

        # The problematic component: FIFOQueue is not serializable by saved_model
        # We keep it here because the original model uses it, but note this won't save properly.
        # Marking as non-trainable state that cannot be serialized.
        # Using tf.queue.FIFOQueue is deprecated in TF2 and causes save errors.
        # To allow eager execution, recreate on call or mark as non-trackable if possible.
        self.default_ids_queue = tf.queue.FIFOQueue(capacity=2_000_000,
                                                   dtypes=[self.key_dtype],
                                                   shapes=[[]],
                                                   name='default_ids_queue')

        self.built = True  # Mark as built

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.int64), tf.TensorSpec(shape=(), dtype=tf.bool)])
    def call(self, ids, training=None):
        # Flatten input ids tensor to 1D for lookup
        flatten_ids = tf.reshape(ids, [-1])
        # Lookup values from the DenseHashTable
        ids_value = self.index_table.lookup(flatten_ids)

        # If training flag is True, enqueue the ids to the FIFOQueue
        if training:
            # Using tf.identity to force dependency on flatten_ids,
            # then enqueue_many to add ids to the queue
            self.update_stat(flatten_ids)

        # Reshape back to original input shape
        ids_index_orig = tf.reshape(ids_value, tf.shape(ids))
        return ids_index_orig

    def update_stat(self, flatten_ids):
        # enqueue_many to add ids to the default_ids_queue; this line
        # causes saved_model serialization error because FIFOQueue resource
        # tensor cannot be saved. This replicates original behavior.
        self.default_ids_queue.enqueue_many(flatten_ids)

    def get_config(self):
        # To enable config saving if needed, though DenseHashTable / FIFOQueue
        # don't serialize fully in saved_model.
        base_config = super().get_config() if hasattr(super(), 'get_config') else {}
        return dict(
            training=self.training,
            key_dtype=self.key_dtype.name if hasattr(self.key_dtype, 'name') else self.key_dtype,
            value_dtype=self.value_dtype.name if hasattr(self.value_dtype, 'name') else self.value_dtype,
            **base_config
        )


def my_model_function():
    return MyModel()


def GetInput():
    # Return a dict with 'a' key containing a batch of 2 int64 tensors shaped [2, 1].
    # Match usage in original example: tf.convert_to_tensor([[2], [200]], dtype=tf.int64)
    input_tensor = tf.random.uniform(shape=(2, 1), minval=0, maxval=300, dtype=tf.int64)
    return {'a': input_tensor}


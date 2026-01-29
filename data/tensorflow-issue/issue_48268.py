# tf.random.uniform((B, 17), dtype=tf.float32) ← Input shape inferred from the original issue's input shape (batch size undefined; feature size = 17)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        # The random shuffle and gather logic moved into a tf.function-compatible approach
        # that avoids indexing issues on TPU by using batch-wise gather with dynamic indices.
        # We keep the feature dimension fixed at 17 as per original example.
        self.data_len = 17

    def call(self, inputs):
        # inputs shape: (batch_size, data_len)
        batch_size = tf.shape(inputs)[0]

        # Create indices [0, 1, ..., data_len-1]
        indices = tf.range(self.data_len, dtype=tf.int32)

        # For each batch element, create a different random permutation of indices.
        # This circumvents the TPU problem with gathering with variable indices slice per batch.
        # We stack batch_size independent shuffles.
        # Using tf.map_fn for batch-wise shuffling:
        def shuffle_one(_):
            return tf.random.shuffle(indices)

        random_indices = tf.map_fn(shuffle_one, tf.range(batch_size), fn_output_signature=tf.int32)
        # random_indices shape: (batch_size, data_len)

        # Gather per-batch inputs according to random_indices:
        # tf.gather supports batch_dims=1 to index each sample independently.
        output = tf.gather(inputs, random_indices, batch_dims=1)

        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random tensor matching expected input (unknown batch size, 17 features)
    # For demonstration, batch size 32 chosen as standard.
    batch_size = 32
    # Here we choose float32 dtype as original data was float32.
    input_tensor = tf.random.uniform((batch_size, 17), dtype=tf.float32)
    return input_tensor


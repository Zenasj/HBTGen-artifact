# tf.random.uniform((1, 25), maxval=100, dtype=tf.int32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Set mixed precision policy as 'mixed_bfloat16'
        # This simulates the user's environment where mixed precision is intended
        policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.set_global_policy(policy)

        # Embedding layer does NOT cast outputs to policy automatically (per reported issue)
        # So its output remains float32 by default
        self.embedding = tf.keras.layers.Embedding(input_dim=100, output_dim=100, 
                                                   dtype=tf.float32,
                                                   name='embedding')

        # Dense layer uses the current policy dtype, i.e. bfloat16
        self.dense = tf.keras.layers.Dense(100, dtype=policy.compute_dtype, name='dense')

        # To avoid the runtime error from mismatched types (float32 + bfloat16),
        # we explicitly cast embedding output to the dense layer output type (bfloat16)
        # This simulates a potential fix/workaround for the reported mixed_precision issue.
        
    def call(self, inputs):
        emb_out = self.embedding(inputs)               # float32 by default
        emb_out_cast = tf.cast(emb_out, dtype=self.dense.dtype)  # Cast to bfloat16

        dense_out = self.dense(tf.cast(inputs, tf.float32))  # inputs need casting for dense
        # Note: original code sums embedding + dense applied to inputs directly.
        # Dense layer expects float inputs, embedding inputs are int32 indices.
        # To replicate user code: dense(inputs) - inputs should be float, so cast here.

        return emb_out_cast + dense_out


def my_model_function():
    # Return an instance of MyModel with mixed precision properly set
    return MyModel()

def GetInput():
    # Return an int32 tensor simulating token IDs for Embedding layer input.
    # Per example, batch=1, sequence length=25, max id=100
    return tf.random.uniform((1, 25), maxval=100, dtype=tf.int32)


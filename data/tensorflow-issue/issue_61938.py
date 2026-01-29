# tf.random.uniform((B,), dtype=tf.int32), tf.random.uniform((1,1,1,1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Embedding with 64 tokens and embedding dimension 128
        self.embedding = tf.keras.layers.Embedding(64, 128)

    @tf.function(jit_compile=True)
    def call(self, x_indices, x_scale):
        # Forward pass does embedding lookup and then element-wise scaling

        # Note: The original issue highlights that when compiled with XLA,
        # out-of-bound indices (e.g., 64) do NOT error out in the embedding layer,
        # whereas uncompiled they do error. This model replicates the example.

        emb = self.embedding(x_indices)  # shape: (B, 128)
        # Broadcast multiply with scaling tensor x_scale (shape e.g. (1,1,1,1))
        # For simplicity: multiply emb by x_scale (broadcast rules apply)
        return emb * x_scale

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return inputs compatible with MyModel
    # input_1: int32 tensor with out-of-bound index 64 to reproduce issue
    # input_2: float32 scaling tensor shaped (1,1,1,1) to broadcast multiply with embedding output
    input_1 = tf.constant([64], dtype=tf.int32)  # Out-of-bound index (valid max is 63)
    input_2 = tf.constant([[[[10.0]]]], dtype=tf.float32)  # Scalar scaling factor with shape for broadcast
    return (input_1, input_2)


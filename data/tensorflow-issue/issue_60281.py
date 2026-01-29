# tf.random.uniform((3, 10, 13), dtype=tf.float32) ← inferred input shape based on provided example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.n_heads = 2
        # WQ shape: (n_heads, feature_dim, 10)
        # According to the issue, shape (2, 13, 10)
        # We will fix the feature dim to 13 per input example shape last dim
        self.WQ = self.add_weight(
            "WQ_encoder", shape=(self.n_heads, 13, 10),
            initializer="random_normal"
        )

    def call(self, inputs, mask=None, *args):
        # Check input shape inference:
        # inputs shape: (batch=3, seq_len=10, feature=13)
        # inputs[:, None, ...] adds axis after batch: shape (3, 1, 10, 13)
        inputs_expanded = inputs[:, None, ...]  # shape (B=3, 1, 10, 13)

        # TensorFlow matmul broadcasting rules prior to some versions on mac M1 cause issues.
        # Here, matmul of inputs_expanded and WQ:
        # inputs_expanded shape: (3, 1, 10, 13)
        # WQ shape: (2, 13, 10)
        # For batching, we need to align dims properly.
        # We'll expand dims of WQ to be (1, n_heads, 13, 10) so broadcast can happen on batch and seq dims

        WQ_expanded = tf.expand_dims(self.WQ, axis=0)  # (1, 2, 13, 10)

        # inputs_expanded shape: (3, 1, 10, 13)
        # We want to use tf.matmul with proper dims:
        # For each batch and head, perform matmul between last two dims of inputs and WQ:
        # inputs_expanded: (3, 1, 10, 13)
        # WQ_expanded: (1, 2, 13, 10)
        # We want to multiply the 13 dim matched:
        # transpose WQ_expanded last two dims to (1, 2, 10, 13) so it can multiply inputs_expanded's last dim 13:

        # However, the original issue expects:
        # matmul(inputs, WQ) along last dimensions so we do:
        # tf.matmul(inputs_expanded, WQ) where inputs_expanded: (..., L, M) and WQ: (..., M, N)

        # So to get (3, 2, 10, 10) output:
        # We need inputs_expanded shape (3, 1, 10, 13)
        # WQ_expanded shape (1, 2, 13, 10)
        # tf.matmul broadcasts on leading dims and multiplies last two dims (10, 13) x (13, 10)
        # but inputs_expanded last two dims are (10, 13), WQ_expanded last two dims are (13, 10)
        # so we need to ensure WQ is shaped appropriately for broadcast

        output = tf.matmul(inputs_expanded, WQ_expanded)
        # output shape: (3, 2, 10, 10)

        # The mask argument is unused as in original example, but kept to maintain signature

        return output

def my_model_function():
    return MyModel()

def GetInput():
    # According to original example usage: np.random.randn(3,10,13)
    # We'll generate a tensor of shape (3, 10, 13) with float32 dtype
    return tf.random.uniform(shape=(3, 10, 13), dtype=tf.float32)


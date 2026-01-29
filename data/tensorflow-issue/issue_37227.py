# tf.random.uniform((BATCH, 64), dtype=tf.float32)
import tensorflow as tf

MINI_BATCH = 3
SPLITS = 2
BATCH = SPLITS * MINI_BATCH


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple linear layer to mimic model behavior from issue
        self.dense = tf.keras.layers.Dense(64, activation=None)

    def call(self, inputs, training=False):
        # A simple forward pass as in example: inputs + 1
        # but using a dense layer instead of '+1' for a model param
        x = self.dense(inputs)
        return x

class CustomLoss(tf.keras.losses.Loss):
    def call(self, labels, outputs):
        # The original bug happened when tf.expand_dims was used inside tf.map_fn on labels
        # The fix is not part of this code, we show the intended logic.
        
        # Wrap expand_dims in tf.function as original code
        @tf.function
        def fxn(inp):
            # Expand dimension 2 of input  
            # Assumption: inp shape is (MINI_BATCH,), expand dim=2 is valid for rank >= 2
            # But the original issue was that inp rank was unknown dynamically,
            # so this replicates the issue context.
            return tf.expand_dims(inp, 2)
        
        # We apply map_fn over labels (assumed shape: [SPLITS, MINI_BATCH, MINI_BATCH])
        mapped = tf.map_fn(fxn, labels, dtype=tf.float32)

        # Return sum of expanded elements as loss scalar
        return tf.reduce_sum(mapped)


def my_model_function():
    return MyModel()

def GetInput():
    # Return a random tensor shape matching the expected input to MyModel
    # From issue: input shape is (BATCH=6, 64)
    return tf.random.uniform((BATCH, 64), dtype=tf.float32)


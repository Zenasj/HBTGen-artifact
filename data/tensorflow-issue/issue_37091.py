# tf.random.uniform((B, 50, 20), dtype=tf.float32) ← Input shape inferred from the example: batch=8, time=50, features=20

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A GRU layer with 100 units followed by a Dense layer outputting 1 unit
        # This matches the minimal example from the issue.
        # Note: To avoid the cudnn kernel issue on GPU, unroll=True would disable cudnn.
        # But here we keep default unroll=False and cudnn usage to reflect the original problem context.
        self.gru = tf.keras.layers.GRU(100)
        self.dense = tf.keras.layers.Dense(1)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        x = self.gru(inputs)
        output = self.dense(x)
        return output

def my_model_function():
    # Return an instance of MyModel.
    # No pretrained weights specified; initialize fresh.
    return MyModel()

def GetInput():
    # Return a random tensor matching expected input shape
    # Batch size=8 (from example), sequence length=50, features=20, float32 as in example
    return tf.random.uniform((8, 50, 20), dtype=tf.float32)


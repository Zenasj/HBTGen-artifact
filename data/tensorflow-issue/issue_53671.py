# tf.random.uniform((B, 1), dtype=tf.float32) ← inferred input shape based on `Input(shape=(1,), dtype=tf.float32)`

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define Dense layers explicitly
        # Two sets of layers for two GPUs, showing split placement
        # Use tf.keras.layers.Dense instead of tf.layers.Dense
        self.dense_gpu0_1 = tf.keras.layers.Dense(10, name="should_be_on_first_gpu_1")
        self.dense_gpu0_2 = tf.keras.layers.Dense(10, name="should_be_on_first_gpu_2")
        self.dense_gpu1_1 = tf.keras.layers.Dense(10, name="should_be_on_second_gpu_1")
        self.dense_gpu1_2 = tf.keras.layers.Dense(10, name="should_be_on_second_gpu_2")

    def call(self, inputs, training=False):
        # inputs: tensor of shape (B, 1) matching Input(shape=(1,))
        # This model simulates executing parts of the model on two different GPUs using 
        # custom device placement within the call.

        # Compute on GPU:0
        with tf.device("/GPU:0"):
            x_gpu0 = self.dense_gpu0_1(inputs)
            x_gpu0 = self.dense_gpu0_2(x_gpu0)

        # Compute on GPU:1
        with tf.device("/GPU:1"):
            x_gpu1 = self.dense_gpu1_1(inputs)
            x_gpu1 = self.dense_gpu1_2(x_gpu1)

        # Note: Actual device placement might still be influenced by TensorFlow runtime
        # but this shows how to colocate the computation explicitly.

        # Output tuple of both results to simulate multi-GPU outputs
        return (x_gpu0, x_gpu1)

def my_model_function():
    # Return an instance of MyModel with all layers initialized
    return MyModel()

def GetInput():
    # Generate a random float32 tensor with batch size 1 and input feature size 1
    # This matches Input(shape=(1,), dtype=tf.float32)
    # Batch size can be arbitrary small batch, e.g. 1
    return tf.random.uniform((1, 1), dtype=tf.float32)


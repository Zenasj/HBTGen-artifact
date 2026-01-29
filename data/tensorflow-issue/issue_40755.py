# tf.random.uniform((N, 10000), dtype=tf.float32) ← Input shape is (batch_size, 10000), float32 tensor

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Nothing to initialize besides possibly device placement
        # Here, we just keep it flexible as in the issue
        
    @tf.function
    def compute(self, tensor):
        # Explicitly place the compute operations on GPU to match fastest path described in issue
        with tf.device("/gpu:0"):
            res = 100 * tensor
            res = tf.signal.rfft(res)  # computationally intensive op
        return res

    @tf.function
    def call(self, inputs):
        # inputs shape: (N, 10000)
        # The model supports batch application by processing tensor slices one-by-one,
        # similar to the direct loop approach that triggers GPU FFT usage.
        # Here, we map compute over dataset slices (dataset logic embedded in tf.function via tf.map_fn),
        # so that the user can apply compute in a batch manner.

        # Using tf.map_fn to simulate "loop" computation on GPU per element, as dataset.map defers to CPU
        return tf.map_fn(self.compute, inputs, fn_output_signature=tf.complex64)

def my_model_function():
    # Return an instance of MyModel (no extra initialization required)
    return MyModel()

def GetInput():
    # Generate a random tensor compatible with the model's expected input,
    # i.e. batch of random float32 tensors with shape (N, 10000)
    N = 100  # batch size chosen per the examples in the issue
    return tf.random.uniform(shape=(N, 10000), dtype=tf.float32)


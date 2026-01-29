# tf.random.uniform((100,), dtype=tf.float32) ← The input is a 1D tensor of length 100 as per input_signature

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # No traditional layers; the forward uses tf.signal.rfft with fft_length = 512

    @tf.function(jit_compile=True)
    def call(self, x):
        # x: [100], perform real FFT with fft_length=512 (padding internally by tf.signal.rfft)
        # This matches the original test function behavior from the issue
        return tf.signal.rfft(x, fft_length=[512])

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor of shape [100], float32 dtype as expected by MyModel
    return tf.random.uniform(shape=(100,), dtype=tf.float32)


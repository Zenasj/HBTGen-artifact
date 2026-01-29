# tf.random.uniform((B, 3), dtype=tf.float32) ← Here B is batch size, feature size is fixed at 3.

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Create a random weight matrix for matmul (3 x 1)
        # To keep consistent shape and behavior, define as a variable
        self.w = tf.Variable(tf.random.uniform((3, 1)), trainable=False)

        # Define the concrete function for the base inference step
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, 3], dtype=tf.float32, name="x")])
        def run(x):
            # Explicit device placement on GPU for matmul + relu
            with tf.device("/GPU:0"):
                return tf.nn.relu(tf.matmul(x, self.w))
        self._run = run
        self._concrete_run = run.get_concrete_function()

        # Define nondifferentiable_batch_function wrapping the concrete run
        # Use batch size parameters from original example: 1 min_batch, 6 max_batch, timeout 10
        # This decorator requires TF 2.1+, closely mirrors example logic in issue
        @tf.nondifferentiable_batch_function(1, 6, 10)
        def batch_run(x):
            return self._concrete_run(x)
        self._batch_run = batch_run

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 3], dtype=tf.float32, name="x")])
    def call(self, x):
        # Use the batch run function on input x
        return self._batch_run(x)

def my_model_function():
    # Instantiate and return MyModel instance
    return MyModel()

def GetInput():
    # Generate input tensor matching expected input shape: (batch_size, 3)
    # Using batch size of 3 as example consistent with original code
    return tf.random.uniform((3, 3), dtype=tf.float32)


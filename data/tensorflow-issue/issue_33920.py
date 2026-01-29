# tf.random.uniform((8,), dtype=tf.float32) ← inferred input shape is a 1D tensor with 8 elements to match `inputs` variable

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original code uses a variable with the input array and applies collective all-reduce across GPUs.
        # We replicate this scenario as a single model with 2 "logical GPUs" simulated within the model.
        
        # Since collective_ops.all_reduce requires a cluster environment and multiple devices,
        # we cannot directly replicate this behavior in a single Keras model in eager mode.
        # Instead, we simulate the pattern: 
        # 1. A variable initialized with input array.
        # 2. Two "sub-tensors" representing replicas (e.g. split input on dim 0).
        # 3. We perform a simulated all-reduce by summing the replicas and dividing by group size.
        
        self.group_size = 4    # as in original code
        self.group_key = 1
        self.instance_key = 1
        self.num_gpus = 2      # inferred from num_gpus_per_node
        self.inputs = tf.constant([0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1], dtype=tf.float32)
        
        # Variable shared by replicas -- same as original "t"
        self.shared_var = tf.Variable(self.inputs, trainable=False, name="shared_var")
        
    def call(self, x):
        # Input x expected to be tensor matching self.inputs shape
        
        # Simulate 2 replicas processing the shared_var independently (like pinned to two GPUs)
        # We split the input into two halves to simulate two replicas
        replica0 = self.shared_var[:4]
        replica1 = self.shared_var[4:]
        
        # Simulate collective all-reduce: sum across all replicas and divide by group size
        # In original code, group_size=4, but only 2 replicas here, simulate 4 devices by scaling
        # We replicate the values to simulate group size 4 reduction. 
        # This is a simplified stand-in for the actual collective all-reduce op.
        all_replicas_sum = replica0 + replica1  # sum over 2, we multiply by 2 to simulate sum over 4 replicas
        simulated_sum = all_replicas_sum * 2   # simulate sum over 4 replicas total
        
        reduced = simulated_sum / self.group_size  # divide by group size to get mean
        
        # Concatenate back to form output that matches input shape (8,)
        output = tf.concat([reduced, reduced], axis=0)
        return output

def my_model_function():
    # Return a fresh instance of MyModel
    return MyModel()

def GetInput():
    # Return a tensor matching the input expected by MyModel call()
    # The input is not actually used in computation in this example because shared_var directly used,
    # but we mimic input shape and dtype from the initial variable inputs.
    return tf.random.uniform((8,), dtype=tf.float32)


# tf.ones((8, 8), dtype=tf.float32) ← Input shape based on the test example in the issue

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Device meshes as numpy arrays - representing 8 GPUs arranged in 2x4 and 4x2 layouts
        self.device_mesh_1 = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        self.device_mesh_2 = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
    
    @tf.function
    def call(self, input_tensor):
        # This mimics the split_helper function from the issue
        
        # The 'xla_sharding' APIs used in the original code are internal and not exposed publicly.
        # Hence, we simulate the concept of mesh splitting and distributed matmul pipeline 
        # with normal TensorFlow ops for this reconstructed example.
        #
        # Since actual xla_sharding.mesh_split is unavailable, these calls cannot be 
        # replicated exactly. So here we implement a computational graph with similar steps.
        #
        # Assumptions: input_tensor is [8,8] float32 tensor
        
        # Step 1: Matmul with an all ones matrix of shape [8,8]
        y_tensor = tf.ones([8, 8], dtype=tf.float32)
        result = tf.matmul(input_tensor, y_tensor)
        
        # Step 2: Simulated "mesh splitting" reshapes and transpose to mimic redistribution
        # Split along batch dim (simulate splitting along device_mesh_1 layout)
        result = tf.reshape(result, [2, 4, 8])  # batch split 8 -> 2 x 4
        result = tf.transpose(result, perm=[1, 0, 2])  # transpose to simulate different shard layout
        result = tf.reshape(result, [8, 8])
        
        # Step 3: Elementwise sqrt (same as math_ops.sqrt)
        result = tf.sqrt(result)
        
        # Step 4: Another simulated mesh splitting and transpose
        result = tf.reshape(result, [4, 2, 8]) # split batch dim 8 -> 4 x 2
        result = tf.transpose(result, perm=[0, 2, 1]) # swap last dims to mimic swapping mesh dims
        result = tf.reshape(result, [4*8, 2]) # reshape to 32 x 2
        
        # Step 5: Final reshape back to [8,8] for output consistency
        result = tf.reshape(result, [8, 8])
        
        return result

def my_model_function():
    # Return an instance of MyModel without any pretrained weights
    return MyModel()

def GetInput():
    # Return a random tensor of shape [8, 8] with dtype float32 matching the example from the issue
    return tf.ones([8, 8], dtype=tf.float32)


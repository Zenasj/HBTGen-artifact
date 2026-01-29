# tf.random.uniform((B, 64, 64, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable weights here; just using tf.nn.space_to_depth and depth_to_space ops.
        # Multiplication by 10 as in the final example from the issue comments.
    
    def call(self, inputs):
        # Following the example with block_size=2 for space_to_depth and depth_to_space.
        # Names inside TF ops do not reflect the user provided names in eager mode,
        # but here we preserve the requested names in call (though actual op names show differently).
        
        # SpaceToDepth operation
        x = tf.nn.space_to_depth(inputs, block_size=2, name="Space2Depth")
        # Multiplication operation
        x = tf.multiply(x, 10.0, name="Multiplication")
        # DepthToSpace operation
        x = tf.nn.depth_to_space(x, block_size=2, name="Depth2Space")
        return x

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor with shape compatible with the model input.
    # Input shape is (64,64,3) as per the issue.
    # Batch dimension can be arbitrary; here we pick 1.
    return tf.random.uniform((1, 64, 64, 3), dtype=tf.float32)


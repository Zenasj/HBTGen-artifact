# tf.random.uniform((1, 256, 256, 3), dtype=tf.float32) ← Input shape inferred from original model input: 256x256 RGB image batch size=1

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Shared Conv2D layer (equivalent to "conv" in original code)
        self.conv = tf.keras.layers.Conv2D(32, kernel_size=1, padding="same")
        
    def call(self, x):
        # Run convolution
        conv_out = self.conv(x)
        
        # MaxPool2D with kernel size 16, stride 1, padding SAME
        max_pool_out = tf.nn.max_pool2d(conv_out, ksize=16, strides=1, padding="SAME")
        
        # Replicates the workaround for CPU run in the original code:
        # 'out_cpu' applies a tf.where condition to force CPU execution 
        cpu_forced_tensor = tf.where(tf.equal(conv_out, conv_out), conv_out, tf.zeros_like(conv_out))
        max_pool_cpu_out = tf.nn.max_pool2d(cpu_forced_tensor, ksize=16, strides=1, padding="SAME")
        
        # Compare outputs of GPU maxpool (max_pool_out) vs CPU maxpool (max_pool_cpu_out)
        # Output a boolean tensor indicating elementwise equality within a small tolerance
        # (using absolute difference < 1e-6 as tolerance)
        diff = tf.abs(max_pool_out - max_pool_cpu_out)
        comparison = diff < 1e-6
        
        # Return the conv output and both maxpool outputs and their comparison
        # Format: (conv_out, max_pool_out, max_pool_cpu_out, comparison_bool)
        return conv_out, max_pool_out, max_pool_cpu_out, comparison

def my_model_function():
    # Return an instance of MyModel; weights are randomly initialized
    return MyModel()

def GetInput():
    # Returns a single random input tensor matching the model input: batch size 1, 256x256 RGB image
    return tf.random.uniform((1, 256, 256, 3), dtype=tf.float32)


# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Based on typical Conv2D images, batch size and width/height inferred from issue example (1, 64, 64, 64)

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model encapsulates the Conv2D + BatchNorm + ReLU block described in the issue.
    It also integrates a fix/workaround for the dilation_rate padding bug discussed,
    which arises when inputs with different spatial sizes are passed to dilated convolutions 
    with padding="SAME".
    
    The core problem was that padded input shapes for the SpaceToBatchND op must be divisible
    by the dilation rate (block size). TensorFlow formerly calculated paddings only once based 
    on the first input shape, causing runtime errors for subsequent inputs with different shapes.
    
    This model includes logic to dynamically pad inputs so their spatial dimensions are divisible 
    by dilation_rate along height and width, avoiding the "padded_shape not divisible by block_shape" error.
    
    Assumptions:
    - Input shape: (batch_size, height, width, channels)
    - Padding mode: "SAME"
    - Model expects 4D inputs of floats.
    
    Note: The original bug was fixed post TF 2.1, but this model shows how you could 
    proactively pad inputs for older/buggy versions or in uncommon scenarios.
    """
    
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding="SAME",
                 dilation_rate=1,
                 use_bias=False,
                 kernel_initializer="he_normal",
                 kernel_regularizer=None,
                 **bn_params):
        super(MyModel, self).__init__()
        
        self.dilation_rate = dilation_rate
        self.padding = padding.upper()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        
        self.conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="VALID",  # We'll apply custom padding manually if needed.
            dilation_rate=self.dilation_rate,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        )
        self.bn = tf.keras.layers.BatchNormalization(**bn_params)
        
    def call(self, inputs, training=None):
        x = inputs
        
        # If padding = SAME and dilation_rate > 1, ensure input height and width are divisible by dilation_rate:
        # Pad inputs manually because TF Conv2D with dilation can fail otherwise on varying input sizes.
        if self.padding == "SAME" and self.dilation_rate > 1:
            # Extract spatial dimensions dynamically (shape can include None batch dim)
            input_shape = tf.shape(x)
            height = input_shape[1]
            width = input_shape[2]
            
            # Compute padding sizes needed to make height and width divisible by dilation_rate
            height_pad = (-height) % self.dilation_rate
            width_pad = (-width) % self.dilation_rate
            
            # Pad only on the right and bottom edges (could be customized)
            pad_top = 0
            pad_left = 0
            pad_bottom = height_pad
            pad_right = width_pad
            
            paddings = [
                [0, 0],  # batch dim no pad
                [pad_top, pad_bottom],
                [pad_left, pad_right],
                [0, 0],  # channels no pad
            ]
            
            # Only apply padding if needed (non-zero)
            if pad_bottom != 0 or pad_right != 0:
                x = tf.pad(x, paddings, mode="CONSTANT")
            
            # Now manually apply "SAME" padding equivalent with VALID conv over padded x
            # Calculate total kernel padding needed for SAME padding for kernel_size and dilation_rate
            kernel_effective = self.kernel_size + (self.kernel_size - 1) * (self.dilation_rate - 1)
            total_padding = kernel_effective - 1
            
            pad_before = total_padding // 2
            pad_after = total_padding - pad_before
            
            paddings_conv = [
                [0, 0],
                [pad_before, pad_after],
                [pad_before, pad_after],
                [0, 0],
            ]
            x = tf.pad(x, paddings_conv, mode="CONSTANT")
        else:
            # For other cases, just use normal padding mode in Conv2D layer
            # (setting padding to SAME would have handled it)
            pass
        
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x

def my_model_function():
    # Return an instance with typical example parameters:
    # filters=64 channels, 1x1 kernel, dilation=6 as in issue repro code
    return MyModel(filters=64, kernel_size=1, dilation_rate=6)

def GetInput():
    # Return a random tensor matching expected input for MyModel:
    # From example: batch=1, height=64, width=64, channels=64
    return tf.random.uniform((1, 64, 64, 64), dtype=tf.float32)


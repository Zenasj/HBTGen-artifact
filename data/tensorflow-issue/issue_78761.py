# tf.random.uniform((B=1, H=28, W=28, C=1), dtype=tf.float32) ← inferred input shape and dtype from sample input

import tensorflow as tf
import warnings

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define layers with mixed precision dtypes as per the example in the issue
        # Conv2DTranspose with float16
        self.conv1 = tf.keras.layers.Conv2DTranspose(
            filters=6,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="valid",
            data_format="channels_last",
            dilation_rate=(1, 1),
            use_bias=True,
            dtype=tf.float16,
            name="conv1_mutated"
        )
        # Flatten with float32
        self.flatten = tf.keras.layers.Flatten(name="tail_flatten", dtype=tf.float32)
        # Dense layer with float64 output
        self.dense = tf.keras.layers.Dense(
            units=10,
            use_bias=True,
            dtype=tf.float64,
            name="tail_fc"
        )

    def call(self, inputs):
        # inputs shape: (batch, 28, 28, 1)
        x = inputs
        # No zero padding applied in the final merged code because it was a no-op ((0,0),(0,0))
        # Apply conv2d transpose layer (float16)
        x = self.conv1(x)
        # The original code transposed conv1 output with an index-based transpose depending on rank:
        # output_transpose = [(0), (0,1), (0,2,1), (0,3,1,2), (0,4,1,2,3)]
        # For 4D tensor, len(conv1_output.shape)-1 == 3, so transpose dims = (0,3,1,2)
        # Apply transpose to move channels to second dimension?
        # Original input shape is (B, H, W, C), conv1 output shape is same rank, so transpose (0,3,1,2)
        x = tf.transpose(x, perm=(0, 3, 1, 2))
        # Flatten output (float32)
        x = tf.cast(x, tf.float32)  # Cast to flatten layer dtype to ensure consistency
        x = self.flatten(x)
        # Dense layer (float64)
        x = tf.cast(x, tf.float64)  # Cast to dense layer input dtype
        x = self.dense(x)
        return x

    def compile(self, enforce_dtype_consistency=False, *args, **kwargs):
        # Custom compile to check for dtype inconsistency in layers and warn user
        dtypes = {layer.dtype for layer in self.layers if hasattr(layer, 'dtype')}
        if len(dtypes) > 1:
            inconsistent_layers = [layer.name for layer in self.layers if hasattr(layer, 'dtype') and layer.dtype in dtypes]
            warning_message = (
                f"Inconsistent dtypes detected in model layers: {dtypes}. "
                "Mixing dtypes (e.g., float16 with float64) can lead to performance issues and numerical instability. "
                "Layers with inconsistent dtypes include: " + ", ".join(inconsistent_layers)
            )
            warnings.warn(warning_message)
            if enforce_dtype_consistency:
                unified_dtype = kwargs.get("dtype", "float32")
                for layer in self.layers:
                    if hasattr(layer, 'dtype') and layer.dtype != unified_dtype:
                        layer.dtype = unified_dtype
                warnings.warn("Dtypes enforced to be consistent across layers with dtype: " + unified_dtype)
        super().compile(*args, **kwargs)

def my_model_function():
    # Return an instance of MyModel; no pretrained weights are provided
    return MyModel()

def GetInput():
    # Return a random tensor input matching expected input shape and dtype:
    # Input shape is (batch=1, height=28, width=28, channels=1)
    # In original example, numpy input is shape [1,1,28,28] transposed to [1,28,28,1]
    # We'll generate a float32 random tensor with that shape.
    return tf.random.uniform((1, 28, 28, 1), dtype=tf.float32)


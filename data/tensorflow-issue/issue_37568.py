# tf.random.uniform((B, 32, 32, 8), dtype=tf.float32)  ← Assumed input shape: batch of 32x32 RGB-like images with 8 channels (based on dataset and label shape)

import tensorflow as tf

height = 5
width = 5
n_filters = 8
output_dim = 32 * 32 * 8  # label dimension per output head

class Conv2d(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1,1), padding='same', activation='relu', name=None):
        super().__init__(name=name)
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, name=f"{name}_conv")
        self.bn = tf.keras.layers.BatchNormalization(scale=False, name=f"{name}_bn")
        self.ac = tf.keras.layers.Activation(activation=activation, name=f"{name}_ac")

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.ac(x)
        return x

class OutputLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
    def call(self, inputs):
        return inputs

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build a grid of Conv2d layers with height*width cells and one shared conv per cell per row/col
        # For each position (h,w), Conv2d layer connected either to inputs (for 0,0), or previous convs in the row/col
        # Output heads are individual OutputLayer layers for each conv output
        self.height = height
        self.width = width
        self.conv_layers = []  # list of list of Conv2d layers
        for h in range(self.height):
            row_layers = []
            for w in range(self.width):
                name = f"conv_{h}_{w}"
                row_layers.append(Conv2d(n_filters, (3,3), name=name))
            self.conv_layers.append(row_layers)
        # Output heads are OutputLayer layers with unique names for each conv output
        self.output_layers = []
        for h in range(self.height):
            for w in range(self.width):
                self.output_layers.append(OutputLayer(name=f"output_{h}_{w}"))

    def call(self, inputs, training=False):
        # Forward pass:
        # For position (h,w):
        #   if h==0 and w==0: apply conv on inputs
        #   elif w==0: apply conv on conv_layers[h-1][w].output
        #   else: apply conv on conv_layers[h][w-1].output
        # Then collect outputs from all conv_layers, pass through OutputLayer,
        # and return as dict with keys "head_conv_{h}_{w}"
        conv_outputs = [[None]*self.width for _ in range(self.height)]
        for h in range(self.height):
            for w in range(self.width):
                if h == 0 and w == 0:
                    x = self.conv_layers[h][w](inputs, training=training)
                elif w == 0:
                    x = self.conv_layers[h][w](conv_outputs[h-1][w], training=training)
                else:
                    x = self.conv_layers[h][w](conv_outputs[h][w-1], training=training)
                conv_outputs[h][w] = x

        outputs = {}
        idx = 0
        for h in range(self.height):
            for w in range(self.width):
                out = self.output_layers[idx](conv_outputs[h][w])
                # Reshape output to (B, output_dim) to match label shape / loss
                # Conv output shape is (B, H, W, C), so flatten spatial dims and channels
                out_flat = tf.reshape(out, [tf.shape(out)[0], -1])
                outputs[f"head_conv_{h}_{w}"] = out_flat
                idx += 1
        return outputs


def my_model_function():
    # Return an instance of MyModel (with no weights loaded by default)
    return MyModel()


def GetInput():
    # Generate a random tensor matching input shape expected by MyModel call.
    # Based on dataset cifar10: shape=(batch_size, height, width, channels)
    # From chunks: batch_size = 8 * 8 = 64 for TPU setting; channels=8 (inferred from label dim = 32*32*8)
    batch_size = 64
    height = 32
    width = 32
    channels = 8  # inferred from output label shape

    # Create uniform random input tensor in [0,1)
    x = tf.random.uniform(
        shape=(batch_size, height, width, channels),
        dtype=tf.float32,
        seed=7777
    )
    return x


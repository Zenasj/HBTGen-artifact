# tf.random.uniform((B, 32*32*8), dtype=tf.float32)

import tensorflow as tf

height = 5
width = 5
n_filters = 8

class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, filters_in, filters_out, name=None):
        super().__init__(name=name)
        self.conv = tf.keras.layers.Conv2D(filters_out, kernel_size=3, padding='same',
                                           activation=None, use_bias=False,
                                           kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.05))
        self.bn = tf.keras.layers.BatchNormalization(scale=False)
        self.activation = tf.keras.layers.ReLU()

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return self.activation(x)

class OutputLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
    def call(self, inputs):
        return inputs

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Create a 5x5 grid of Conv layers arranged similarly to example
        # Input channels start at 3, all subsequent have n_filters input
        # Each conv layer output channels = n_filters=8
        self.height = height
        self.width = width
        self.n_filters = n_filters
        
        self.layers_grid = []
        for h in range(self.height):
            row_layers = []
            for w in range(self.width):
                if h == 0 and w == 0:
                    filters_in = 3
                elif w == 0:
                    filters_in = n_filters
                else:
                    filters_in = n_filters
                layer = ConvLayer(filters_in, n_filters, name=f"conv_{h}_{w}")
                row_layers.append(layer)
            self.layers_grid.append(row_layers)
        # Output layer is identity
        self.output_layer = OutputLayer(name="output")

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        # inputs shape is (batch, H_in, W_in, 3)
        batch_size = tf.shape(inputs)[0]
        seqs = []
        for h in range(self.height):
            seq = []
            for w in range(self.width):
                if h == 0 and w == 0:
                    x = inputs
                elif w == 0:
                    x = seqs[h-1][0]  # conv at previous height, same width 0
                else:
                    x = seq[w-1]  # previous conv in same row
                conv_layer = self.layers_grid[h][w]
                y = conv_layer(x, training=training)
                seq.append(y)
            seqs.append(seq)
        # For this example, select the first conv output (0,0) to produce logits
        # Flatten last conv output of conv_0_0 to shape (batch, 32*32*8)
        tensor = seqs[0][0]  # conv_0_0 output tensor
        # Assuming inputs from CIFAR10: 32x32 spatial dims
        batch, h_in, w_in, channels = tf.shape(tensor)[0], tf.shape(tensor)[1], tf.shape(tensor)[2], tf.shape(tensor)[3]
        tensor_flat = tf.reshape(tensor, [batch, -1], name="reshaped_head_conv_0_0")
        return tensor_flat

def my_model_function():
    # Return an instance of MyModel, no pretrained weights specified
    return MyModel()

def GetInput():
    # Return a random input tensor that matches input expected by MyModel
    # MyModel expects (batch, 32, 32, 3) images (like CIFAR10 format)
    batch_size = 8
    return tf.random.uniform(shape=(batch_size, 32, 32, 3), dtype=tf.float32)


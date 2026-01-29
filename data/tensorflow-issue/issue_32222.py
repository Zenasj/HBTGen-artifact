# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Assuming 4D input for ConcatCoords4D, typical conv input shape: batch, height, width, channels
import tensorflow as tf

def normalize(x, epsilon=1e-7):
    """Normalize a tensor to range [0,1]."""
    x_min = tf.reduce_min(x)
    x_max = tf.reduce_max(x)
    return (x - x_min) / (x_max - x_min + epsilon)

class ConcatCoords2D(tf.keras.layers.Layer):
    def __init__(self):
        super(ConcatCoords2D, self).__init__()
        # Internally use 3D coordinate concatenation by expanding dims
        self.handler = ConcatCoords3D()
        self.built = True

    @tf.function
    def call(self, x):
        # Expand dims to make input 3D: shape (batch, length, channels) -> (batch, length, 1, channels)
        x_exp = tf.expand_dims(x, -2)  # Insert dimension for length in middle
        # The original code expands dims differently, but here input assumed to be 3D tensor: (batch, length, channels)
        # To make it consistent, let's expand -1 dim, then apply ConcatCoords3D
        # For simplicity, adapting to expand last dim
        return self.handler(x_exp)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape.append(1)
        output_shape[-1] = output_shape[-1] + 1
        return tuple(output_shape)

class ConcatCoords3D(tf.keras.layers.Layer):
    def __init__(self):
        super(ConcatCoords3D, self).__init__()
        self.built = True

    @tf.function
    def call(self, x):
        # x is 3D tensor: (batch_size, length, channels)
        shape = tf.shape(x)
        # Coordinates along length dimension
        coords = tf.range(shape[1], dtype=tf.float32)  # length dimension
        coords = tf.expand_dims(coords, 0)  # shape (1, length)
        coords = tf.expand_dims(coords, -1) # shape (1, length, 1)
        coords = tf.tile(coords, [shape[0], 1, 1]) # (batch_size, length, 1)
        coords = normalize(coords)
        return tf.concat([x, coords], axis=-1)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = output_shape[-1] + 1
        return tuple(output_shape)

class ConcatCoords4D(tf.keras.layers.Layer):
    def __init__(self):
        super(ConcatCoords4D, self).__init__()
        self.hw = None

    def build(self, input_shape):
        # input_shape is (batch, height, width, channels)
        h = tf.range(input_shape[1], dtype=tf.float32)
        w = tf.range(input_shape[2], dtype=tf.float32)
        h = normalize(h)
        w = normalize(w)
        # Create a meshgrid from normalized height and width values
        meshgrid = tf.stack(tf.meshgrid(h, w, indexing='ij'), axis=-1)  # shape (height, width, 2)
        meshgrid = tf.expand_dims(meshgrid, 0)  # shape (1, height, width, 2)
        self.hw = tf.tile(meshgrid, [input_shape[0], 1, 1, 1])  # shape (batch, height, width, 2)
        super(ConcatCoords4D, self).build(input_shape)

    @tf.function
    def call(self, x):
        # Concatenate coordinate channels to input tensor along channels axis
        return tf.concat([x, self.hw], axis=-1)  # Adds 2 channels

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = output_shape[-1] + 2
        return tuple(output_shape)

class MyModel(tf.keras.Model):
    """
    MyModel performs CoordConv-style coordinate concatenation on input tensors.
    It delegates to appropriate CoordConcat layers depending on input rank.

    - For 2D inputs (batch, length), uses ConcatCoords2D
    - For 3D inputs (batch, length, channels), uses ConcatCoords3D
    - For 4D inputs (batch, height, width, channels), uses ConcatCoords4D
    """
    def __init__(self):
        super(MyModel, self).__init__()
        # Initialize possible coordinate concat layers
        self.concat2d = ConcatCoords2D()
        self.concat3d = ConcatCoords3D()
        self.concat4d = ConcatCoords4D()

    @tf.function(jit_compile=True)
    def call(self, x):
        rank = len(x.shape)
        if rank == 2:
            # Input shape: (batch, length)
            return self.concat2d(x)
        elif rank == 3:
            # Input shape: (batch, length, channels)
            return self.concat3d(x)
        elif rank == 4:
            # Input shape: (batch, height, width, channels)
            return self.concat4d(x)
        else:
            raise ValueError(f"Input rank {rank} not supported by MyModel; "
                             f"expected 2, 3, or 4")

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random 4D tensor typical for convolutional input:
    # For example, batch size 2, height 28, width 28, channels 3
    batch_size = 2
    height = 28
    width = 28
    channels = 3
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)


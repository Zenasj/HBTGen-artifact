# tf.random.uniform((B, 2, 2), dtype=tf.int32) ← Based on input shape in example: Input shape=(None, 2, 2) of dtype int32

import tensorflow as tf

class Lookup(tf.keras.layers.Layer):
    def __init__(self, depth: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.lookup_table = None

    def build(self, input_shape):
        # input_shape: (batch_size, 2, 2)
        # We want the lookup table shape to match the last dimension of the indices used for gather_nd.
        # The gather indices have shape (batch, 2, 2), so indices shape[-1] == 2.
        # From the example, the lookup_table shape is input_shape[1:-1] + [self.depth]
        # input_shape[1:-1] == (2,), so lookup_table shape = (2, depth)
        # This means "params" tensor's rank=2, so indices last dim=2 <= params rank=2 => valid.
        lookup_shape = tuple(input_shape[1:-1]) + (self.depth,)
        self.lookup_table = self.add_weight(
            name='lookup_table',
            shape=lookup_shape,
            dtype='float32',
            initializer='random_normal',
            trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs: int32 indices with shape (batch, 2, 2)
        # gather_nd params: shape (2, depth)
        # gather_nd indices shape last dimension = 2 (matches rank of params)
        # returns entries: shape (batch, 2, depth)
        entries = tf.gather_nd(params=self.lookup_table, indices=inputs, name='lookup_call')
        return entries

    def get_config(self):
        config = super().get_config()
        config.update({"depth": self.depth})
        return config

class MyModel(tf.keras.Model):
    def __init__(self, depth=3, **kwargs):
        super().__init__(**kwargs)
        # Instantiate the custom lookup layer
        self.lookup = Lookup(depth=depth)

    def call(self, inputs, **kwargs):
        # Forward pass simply applies the Lookup layer
        return self.lookup(inputs)

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random int32 tensor matching input shape expected by MyModel
    # Input shape is (batch_size, 2, 2) of dtype int32.
    # 
    # Indices must be valid to index into lookup_table of shape (2, depth).
    # So indices values must be in [0, 1] for first index dimension (max 2),
    # and in [0, depth-1] for second index dimension. But inputs shape last dim=2,
    # so the indices themselves have shape (2,), meaning each index points to a single position in lookup_table.
    #
    # However, gather_nd indices shape is (batch, 2, 2), where the last dim=2 means the indices are pairs of indices.
    # Valid index values are from 0 to the corresponding dimension - 1 of lookup_table.
    #
    # So the first element of the pair is in [0, 1], second element in [0, depth-1].
    # To simplify: since lookup_table shape is (2, depth), indices' last dim=2 means
    # indices like [x, y] where x in [0,1], y in [0,depth-1]

    batch_size = 4
    depth = 3  # keep consistent with default depth
    # We generate random indices where:
    # indices[..., 0] in [0, 1]
    # indices[..., 1] in [0, depth-1]
    indices_dim0 = tf.random.uniform((batch_size, 2, 1), minval=0, maxval=2, dtype=tf.int32)
    indices_dim1 = tf.random.uniform((batch_size, 2, 1), minval=0, maxval=depth, dtype=tf.int32)
    indices = tf.concat([indices_dim0, indices_dim1], axis=-1)  # final shape (batch_size, 2, 2)
    return indices


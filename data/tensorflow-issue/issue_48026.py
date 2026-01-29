# tf.random.uniform((2, 4, 3, 1), dtype=tf.float32) ← Input shape inferred from mini_batch in the issue: (batch=2, n_time_steps=4, n_features=3, n_channels=1)
import tensorflow as tf

class TimeDistributedMaskPropagating(tf.keras.layers.TimeDistributed):
    """TimeDistributed layer that propagates mask."""
    
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)
        self.supports_masking = True
        
    def compute_mask(self, inputs, mask=None):
        # Propagate mask unchanged
        return mask

class MyModel(tf.keras.Model):
    def __init__(self, n_features=3, n_channels=1):
        super().__init__()
        # Masking layer -- the default Masking works dimension-wise along last axis (axis=-1),
        # but here the input is rank 4 and mask computed ends up rank 3.
        # To fix: override compute_mask to aggregate over all feature dims except batch & time.
        # Instead of patching 'Masking', recreate the logic here to give rank 2 mask: (batch, time).
        # We'll define a custom Masking layer that masks timesteps if all values along axes [2, 3] equal mask_value.
        self.masking = CustomMasking(mask_value=0.0)

        # TimeDistributed with a simple cnn_block (Flatten)
        cnn_block = tf.keras.layers.Flatten()
        self.time_dist = TimeDistributedMaskPropagating(cnn_block)

        # RNN layer expects mask shape (batch, time)
        self.lstm = tf.keras.layers.LSTM(10)

    def call(self, inputs, training=None):
        x = self.masking(inputs)
        x = self.time_dist(x)
        # The masking layer returns a rank 2 mask; it will be passed automatically to LSTM
        output = self.lstm(x)
        return output

class CustomMasking(tf.keras.layers.Layer):
    """
    Custom masking layer that computes mask over timesteps (axis=1),
    masking any timestep where all values along axes 2..rank are equal to mask_value.
    This addresses the issue described: Masking from tf.keras.layers.Masking
    produces mask of rank equal to inputs - 1, but here we want rank 2 mask (batch, time).
    """
    def __init__(self, mask_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.mask_value = mask_value
        self.supports_masking = True

    def call(self, inputs):
        # Simply return inputs (pass-through)
        return inputs

    def compute_mask(self, inputs, mask=None):
        # inputs shape: (batch, time, ...)
        # Mask if all features/channels == mask_value at that timestep
        # Reduce over all dims except batch & time (axes 2 and onwards)
        # Result mask shape: (batch, time)
        # Use tf.reduce_all to combine masks along those axes
        # Example: mask = tf.reduce_all(tf.equal(inputs, mask_value), axis=[2, 3, ...])
        # But generalize to arbitrary rank inputs (rank >=3)
        input_shape = tf.shape(inputs)
        rank = inputs.shape.rank
        # Defensive: if rank < 3, fallback to original masking behavior (mask over last axis)
        if rank is None or rank < 3:
            # fallback: mask over last axis
            return tf.reduce_all(tf.equal(inputs, self.mask_value), axis=-1)

        axes_to_reduce = list(range(2, rank))
        mask = tf.reduce_all(tf.equal(inputs, self.mask_value), axis=axes_to_reduce)
        return mask

def my_model_function():
    # Return an instance of MyModel with default dimensions
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the expected input shape:
    # batch_size = 2, n_time_steps = 4, n_features=3, n_channels=1, dtype float32
    # We'll generate inputs with zeros padded as in the minimal example:
    # First sample: shape (4, 3, 1) random uniform
    x1 = tf.random.uniform((4, 3, 1), dtype=tf.float32)
    # Second sample: shape (3, 3, 1) random uniform padded to (4, 3, 1)
    x2 = tf.random.uniform((3, 3, 1), dtype=tf.float32)
    paddings = tf.constant([[0, 1], [0, 0], [0, 0]])
    padded_x2 = tf.pad(x2, paddings)
    # Stack to batch size 2: shape (2, 4, 3, 1)
    batch = tf.stack((x1, padded_x2))
    return batch


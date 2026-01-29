# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Input assumed to be 4D tensor with channels_last, typical for image tensors

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.util.tf_export import keras_export

@keras_export('keras.constraints.UnitSumNonNeg', 'keras.constraints.unit_sum_non_neg')
class UnitSumNonNeg(tf.keras.constraints.Constraint):
    """
    Constraint to ensure weights are non-negative and sum to one along axis 0.
    
    Used to maintain trainable kernel in OWA pooling layer.
    """
    def __call__(self, w):
        # Clamp weights to non-negative values
        aux =  w * tf.cast(tf.math.greater_equal(w, 0.), w.dtype)
        # Normalize so weights along axis 0 sum to one
        return aux / (K.epsilon() + tf.reduce_sum(aux, axis=[0], keepdims=True))


class OWAPoolingNew(tf.keras.layers.Layer):
    """
    Custom OWA (Ordered Weighted Averaging) Pooling Layer.

    This layer extracts image patches (like pooling windows), sorts them
    (descending order) if enabled, and applies weighted sum pooling with trainable weights
    constrained to be non-negative and sum to one.

    Parameters:
      pool_size: tuple of 2 ints, window size
      strides: tuple of 2 ints, stride size (defaults to pool_size)
      padding: 'valid' or 'same' (currently used 'same' in call)
      data_format: channels_last assumed and forced
      sort: whether to sort patches before weighted sum
      train: whether kernel weights are trainable
      seed: random seed for weight initialization
      all_channels: if True, generates separate weights per channel
    """
    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding='valid',
        data_format=None,
        name=None,
        sort=True,
        train=True,
        seed=None,
        all_channels=False,
        **kwargs):
        super(OWAPoolingNew, self).__init__(name=name, **kwargs)

        self.pool_size = pool_size
        # Default strides to pool_size if not specified
        self.strides = pool_size if strides is None else strides
        self.padding = padding
        # Force channels_last data format for simplicity
        self.data_format = conv_utils.normalize_data_format('channels_last')
        self.sort = sort
        self.train = train
        self.seed = seed if seed is not None else 10
        self.all_channels = all_channels
        
    def build(self, input_shape):
        # input_shape: (batch, height, width, channels)
        channels = input_shape[-1]

        # Determine kernel weight shape:
        # If all_channels=True: one weight vector per channel,
        # else single vector broadcast across channels.
        if self.all_channels:
            weights_shape = (self.pool_size[0] * self.pool_size[1], channels)
        else:
            weights_shape = (self.pool_size[0] * self.pool_size[1], 1)
        
        tf.random.set_seed(self.seed)
        # Initialize kernel weights uniformly and normalize to sum to 1
        kernel = tf.random.uniform(shape=weights_shape, dtype=tf.float32)
        kernel /= tf.reduce_sum(kernel, axis=[0], keepdims=True)
        # Create kernel variable trainable according to flag, with UnitSumNonNeg constraint
        self.kernel = self.add_weight(
            name='kernel',
            shape=weights_shape,
            initializer=tf.constant_initializer(kernel.numpy()),  # fixed initial value
            trainable=self.train,
            dtype=tf.float32,
            constraint=UnitSumNonNeg()
        )

    def call(self, inputs):
        # inputs shape: [batch, height, width, channels]

        # Extract input dimensions
        batch_size, height, width, channels = tf.unstack(tf.shape(inputs))
        # Use static shape as fallback for static dims
        static_shape = inputs.get_shape().as_list()
        if static_shape[1] is not None:
            height = static_shape[1]
        if static_shape[2] is not None:
            width = static_shape[2]
        if static_shape[3] is not None:
            channels = static_shape[3]

        # Prepare parameters for tf.image.extract_patches
        ksize = [1, self.pool_size[0], self.pool_size[1], 1]
        stride = [1, self.strides[0], self.strides[1], 1]

        # Extract patches from input, shape: [batch, new_h, new_w, patch_size * channels]
        patches = tf.image.extract_patches(
            inputs,
            sizes=ksize,
            strides=stride,
            rates=[1, 1, 1, 1],
            padding='SAME'  # The original code hardcoded SAME padding here
        )
        # patches shape: [batch, pool_height, pool_width, patch_elements]
        # patch_elements == pool_size[0]*pool_size[1]*channels

        # Number of elements per patch per channel
        patch_elems_total = patches.shape[-1]
        elems_per_channel = (self.pool_size[0] * self.pool_size[1])

        # Infer pool height and pool width
        pool_height = patches.shape[1]
        pool_width = patches.shape[2]

        # Reshape to separate channels dimension
        # patches: [batch, pool_height, pool_width, patch_size*channels]
        # reshape to [batch, pool_height, pool_width, patch_size, channels]
        patches_reshaped = tf.reshape(
            patches,
            [-1, pool_height, pool_width, elems_per_channel, channels]
        )

        # If sorting enabled, sort patches descending on patch dimension (last-but-one)
        if self.sort:
            patches_sorted = tf.sort(patches_reshaped, axis=3, direction='DESCENDING')
        else:
            patches_sorted = patches_reshaped

        # Weighted sum pooling with kernel weights:
        # kernel shape: [patch_size, channels] or [patch_size, 1]
        # We multiply patches by kernel weights per element and sum along patch_size axis (axis=3)
        outputs = tf.reduce_sum(patches_sorted * self.kernel, axis=3)

        # Result shape: [batch, pool_height, pool_width, channels]
        return outputs


class MyModel(tf.keras.Model):
    """
    Combined model class wrapping the custom OWAPoolingNew layer alongside
    a standard MaxPooling2D layer to compare outputs.

    The forward pass returns a dictionary containing:
    - 'owa': output from OWAPoolingNew
    - 'max': output from MaxPooling2D
    - 'close': boolean tensor indicating element-wise closeness within tolerance

    This model allows comparing performance and numerical proximity of the two pooling methods.
    """

    def __init__(self, pool_size=(2, 2), strides=None, padding='valid', sort=True, train=True, all_channels=False):
        super(MyModel, self).__init__()

        # Initialize the custom OWA pooling layer
        self.owa_pool = OWAPoolingNew(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            sort=sort,
            train=train,
            all_channels=all_channels
        )

        # Initialize standard MaxPooling2D for comparison
        # Note: For padding consistency, map 'valid' or 'same' string
        self.max_pool = tf.keras.layers.MaxPooling2D(
            pool_size=pool_size,
            strides=strides if strides is not None else pool_size,
            padding=padding
        )

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # Compute OWA pooling output
        owa_out = self.owa_pool(inputs)
        # Compute MaxPooling2D output
        max_out = self.max_pool(inputs)

        # Compare outputs element-wise within tolerance
        # Here tolerance is abs diff <= 1e-5 (can be adjusted)
        is_close = tf.math.less_equal(tf.abs(owa_out - max_out), 1e-5)

        # Return dictionary with both outputs and boolean comparison tensor
        return {
            'owa': owa_out,
            'max': max_out,
            'close': is_close
        }


def my_model_function():
    """
    Returns:
      An instance of MyModel with default parameters.
      This model wraps OWAPoolingNew and MaxPooling2D and compares outputs.
    """
    return MyModel()


def GetInput():
    """
    Returns a random input tensor matching the expected input shape and dtype for MyModel.
    
    Chooses a batch size of 4, height and width 32, channels 3 to reflect typical image input.
    Values are in [0,1) float32 as typical for image data preprocessing.
    """
    batch = 4
    height = 32
    width = 32
    channels = 3
    return tf.random.uniform(shape=(batch, height, width, channels), dtype=tf.float32)


# tf.random.uniform((3, 5, 1, 1), dtype=tf.float64)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We instantiate the three layers discussed in the issue
        self.global_max_pool_2d = tf.keras.layers.GlobalMaxPooling2D(
            data_format="channels_last", keepdims=False
        )
        # For completeness and fusion, instantiate 3D and 1D pooling layers as well
        self.global_max_pool_3d = tf.keras.layers.GlobalMaxPooling3D(
            data_format="channels_first", keepdims=False
        )
        self.max_pool_1d = tf.keras.layers.MaxPooling1D(
            pool_size=2, strides=3, padding="valid", data_format="channels_last"
        )
    
    def call(self, inputs):
        """
        Unified call method that demonstrates the pooling layers used in the issue.
        Since inputs shape differs per layer in the original examples,
        here we will expect a dictionary of inputs for each layer keyed by:
          '2d', '3d', '1d'
        and produce a dict of output tensors after applying respective pools.

        We also compute a simple comparison metric for each layer's
        gradient inconsistency example by returning difference between
        theoretical and numerical gradient placeholder values,
        emulating the "comparison" logic mentioned in the issue.
        
        Note: Since gradient computation is done outside the forward pass,
        here we simply demonstrate running the layers and returning output
        for the provided inputs. Gradient comparison logic should be implemented
        externally using tf.test.compute_gradient.
        """
        # Extract inputs for each pooling type
        input_2d = inputs.get('2d')    # expected shape: (3, 5, 1, 1) for GlobalMaxPooling2D
        input_3d = inputs.get('3d')    # expected shape: (2, 4, 3, 1, 1) for GlobalMaxPooling3D
        input_1d = inputs.get('1d')    # expected shape: (3, 5, 4) for MaxPooling1D

        outputs = {}

        if input_2d is not None:
            outputs['2d'] = self.global_max_pool_2d(input_2d)
        if input_3d is not None:
            outputs['3d'] = self.global_max_pool_3d(input_3d)
        if input_1d is not None:
            outputs['1d'] = self.max_pool_1d(input_1d)

        return outputs

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    """
    Generate a dictionary of inputs matching expected shapes and dtypes used
    in the issue's examples for each pooling type.

    The inputs are chosen to be consistent with the bug reproduction code:
    - 2D GlobalMaxPooling2D: shape (3,5,1,1), dtype float64, channels_last
    - 3D GlobalMaxPooling3D: shape (2,4,3,1,1), dtype float64, channels_first
    - 1D MaxPooling1D: shape (3,5,4), dtype float64, channels_last
    """
    input_2d = tf.random.uniform(
        [3, 5, 1, 1], minval=0, maxval=1, dtype=tf.float64
    )
    input_3d = tf.random.uniform(
        [2, 4, 3, 1, 1], minval=0, maxval=1, dtype=tf.float64
    )
    input_1d = tf.random.uniform(
        [3, 5, 4], minval=0, maxval=1, dtype=tf.float64
    )
    return {
        '2d': input_2d,
        '3d': input_3d,
        '1d': input_1d,
    }


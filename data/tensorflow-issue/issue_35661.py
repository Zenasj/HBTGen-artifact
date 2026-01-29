# tf.random.uniform((B, 35, 384), dtype=tf.float32) and tf.random.uniform((B, 35), dtype=tf.int32)
import tensorflow as tf
import numpy as np

class RepeatLayers(tf.keras.layers.Layer):
    def __init__(self, axis=0):
        super(RepeatLayers, self).__init__()
        self.axis = axis

    def _all_dimensions(self, x):
        # Return tensor with range [0, rank(x)), used for reduction dimensions
        if isinstance(x, tf.Tensor) and x.shape.ndims is not None:
            return tf.constant(np.arange(x.shape.ndims), dtype=tf.int32)
        # Fallback for SparseTensor or unknown rank - generate range dynamically
        return tf.range(tf.rank(x))
    
    def _tile_one_dimension(self, data, axis, multiple):
        # Tile data along specified axis by 'multiple'
        if data.shape.ndims is not None:
            multiples = [1] * data.shape.ndims
            multiples[axis] = multiple
        else:
            ones_value = tf.ones(tf.rank(data), tf.int32)
            multiples = tf.concat([
                ones_value[:axis], 
                [multiple], 
                ones_value[axis + 1:]],
                axis=0)
        return tf.tile(data, multiples)

    def repeat_with_axis(self, data, repeats, axis):
        # Repeat slices of data along axis by repeat counts in repeats
        # data shape example: [B, max_len, d], repeats shape: [B, max_len]
        data = tf.convert_to_tensor(data, name='data')
        repeats = tf.cast(tf.convert_to_tensor(repeats, name='repeats'), tf.int32)

        data_shape = tf.shape(data)  # dynamic shape

        # To avoid errors during TFLite conversion, ensure max_repeat >= 1 (see issue discussion)
        max_repeat = tf.math.maximum(
            1,
            tf.reduce_max(repeats, axis=self._all_dimensions(repeats))
        )  # scalar

        mask = tf.sequence_mask(repeats, maxlen=max_repeat)  # [B, max_len, max_repeat], bool

        expanded = tf.expand_dims(data, axis + 1)  # [B, max_len, 1, d]
        tiled = self._tile_one_dimension(expanded, axis + 1, max_repeat)  # [B, max_len, max_repeat, d]

        masked = tf.boolean_mask(tiled, mask)  # flatten masked elements
        # Construct new shape:
        # combine dimensions before axis,
        # flatten repeated dimension (-1),
        # then dimensions after axis
        # Cast intermediate parts to int32 to avoid concat type mismatch (issue discussion)
        prefix_shape = tf.cast(data_shape[:axis], tf.int32)
        suffix_shape = tf.cast(data_shape[axis + 1:], tf.int32)
        result_shape = tf.concat([prefix_shape, [-1], suffix_shape], axis=0)
        result = tf.reshape(masked, result_shape)

        return result

    def call(self, encoder_h, repeats):
        return self.repeat_with_axis(data=encoder_h, repeats=repeats, axis=self.axis)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.repeat_layer = RepeatLayers(axis=1)

    def call(self, inputs):
        encoder_h, repeats = inputs
        return self.repeat_layer(encoder_h, repeats)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate sample inputs consistent with the expected model input shapes:
    # encoder_h shape: [B, 35, 384], float32
    # repeats shape: [B, 35], int32 with small integer values >=1 to avoid TFLite range errors
    B = 2  # batch size as example
    encoder_h = tf.random.uniform((B, 35, 384), dtype=tf.float32)

    # Generate repeats with at least 1 to avoid issues with range ops in TFLite
    # Let's use small integers in range [1, 5]
    repeats = tf.random.uniform((B, 35), minval=1, maxval=6, dtype=tf.int32)

    return (encoder_h, repeats)


# tf.random.uniform((1, 16, 16, 1), dtype=tf.float32) ← Input shape inferred from TEST_SIZE=16, batch 1, 1 channel

import tensorflow as tf
from tensorflow import keras

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.average_pool = keras.layers.AveragePooling2D((6, 6), padding='same', strides=4)
        self.unaverage_pool = UnAveragePooling2D(kernel=(6,6), strides=4)

    def call(self, inputs):
        x = self.average_pool(inputs)
        x = self.unaverage_pool(x)
        return x

def my_model_function():
    # Return an instance of MyModel, with all required layers initialized
    return MyModel()

def GetInput():
    # Generate a random float32 input tensor matching the expected input shape of (1, 16, 16, 1)
    return tf.random.uniform((1, 16, 16, 1), dtype=tf.float32)


class UnAveragePooling2D(keras.layers.Layer):
    def __init__(self, kernel, *, strides, name='co1', dtype=None):
        super().__init__(trainable=True, name=name, dtype=dtype)
        self.kernel = kernel
        self.strides = strides

    def build(self, input_shape):
        # No weights or variables to create
        pass

    def call(self, inputs):
        inputs_shape = tf.shape(inputs)

        # The averaged image was at least source.shape * strides, assume multiple of stride as in original code.
        dest_shape = (inputs_shape[0], inputs.shape[1] * self.strides, inputs.shape[2] * self.strides, inputs_shape[3])
        
        # Width of destination invalid edges to be filled (assume half the stride)
        dest_start_invalid = (self.strides // 2, self.strides // 2)
        dest_end_invalid = (self.strides // 2, self.strides // 2)

        strides_f = tf.cast(self.strides, tf.float32)
        dest_start_invalid_f = (tf.cast(dest_start_invalid[0], tf.float32), tf.cast(dest_start_invalid[1], tf.float32))
        max_source_f = (tf.cast(inputs.shape[1] - 1, tf.float32), tf.cast(inputs.shape[2] - 1, tf.float32))

        # Build output tensor by batching rows and columns with bilinear interpolation.
        dest = tf.zeros((dest_shape[0], 0, dest_shape[2], dest_shape[3]), dtype=inputs.dtype)
        for dest_r in range(dest_shape[1]):
            dest_r_f = tf.cast(dest_r, tf.float32)
            row = tf.zeros((dest_shape[0], 1, 0, dest_shape[3]), dtype=inputs.dtype)
            for dest_c in range(dest_shape[2]):
                dest_c_f = tf.cast(dest_c, tf.float32)
                source_r = self._dest_to_rource(dest_r_f, strides_f, dest_start_invalid_f[0], max_source_f[0])
                source_c = self._dest_to_rource(dest_c_f, strides_f, dest_start_invalid_f[1], max_source_f[1])
                value = self._bilinear_interpolate(inputs, source_r, source_c)
                value = tf.reshape(value, (inputs_shape[0], 1, 1, inputs_shape[3]))
                row = tf.concat([row, value], axis=2)
            dest = tf.concat([dest, row], axis=1)

        return dest

    @tf.function
    def _dest_to_rource(self, dest, stride, dest_start_invalid, max_source):
        """
        Given a destination pixel row or column position (float),
        compute the corresponding source coordinate to interpolate from.
        """
        if dest < dest_start_invalid + stride - 0.5:
            return (dest - dest_start_invalid) / (stride - 0.5)
        elif dest > dest_start_invalid + (max_source - 1.0) * stride - 0.5:
            return ((dest - dest_start_invalid + 0.5) - (max_source - 1.0) * stride) / (stride - 0.5) + max_source - 1.0
        else:
            return (dest - dest_start_invalid + 0.5) / stride

    @tf.function
    def _bilinear_interpolate(self, source, r, c):
        """
        Bilinear interpolate the batch of images `source` at fractional coordinates (r, c).
        Uses pixels around floored indices; fades out near edges by using zero padding.
        """
        r0 = tf.cast(tf.floor(r), tf.int32)
        c0 = tf.cast(tf.floor(c), tf.int32)
        fr = r - tf.cast(r0, tf.float32)
        fc = c - tf.cast(c0, tf.float32)

        p00 = self._safe_lookup_pixel(source, r0, c0)
        p01 = self._safe_lookup_pixel(source, r0, c0 + 1)
        p10 = self._safe_lookup_pixel(source, r0 + 1, c0)
        p11 = self._safe_lookup_pixel(source, r0 + 1, c0 + 1)

        return p00 * (1 - fr) * (1 - fc) + p01 * (1 - fr) * fc + p10 * fr * (1 - fc) + p11 * fr * fc

    @tf.function
    def _safe_lookup_pixel(self, source, r, c):
        """
        Look up a batch of pixels at indices r, c in the source tensor.
        If indices are out of bounds, return zeros (black pixels).
        """
        # Use TensorFlow ops to check bounds, then choose pixels or zeros per batch.
        source_shape = tf.shape(source)
        batch_size = source_shape[0]
        height = source_shape[1]
        width = source_shape[2]
        channels = source_shape[3]

        def get_pixels():
            # Indices are scalar ints, gather pixels from each batch
            batch_indices = tf.range(batch_size, dtype=tf.int32)
            # Gather pixels at (batch, r, c, :) using tf.gather_nd
            indices = tf.stack([batch_indices,
                                tf.fill([batch_size], r),
                                tf.fill([batch_size], c)], axis=1)
            return tf.gather_nd(source, indices)

        # Check if r and c are in bounds (r < height, c < width and >= 0)
        cond = tf.logical_and(
            tf.logical_and(r >= 0, r < height),
            tf.logical_and(c >= 0, c < width)
        )

        # Using tf.cond inside @tf.function on scalars can cause XLA issues.
        # Instead implement safe pixel lookup using boolean masking in graph-compatible way:
        # We do bounds check per scalar (r,c) so it is constant per call.
        # So we do:
        # If cond True: gather pixels per batch
        # Else: return zeros of shape (batch_size, channels).
        return tf.cond(cond,
                       true_fn=get_pixels,
                       false_fn=lambda: tf.zeros((batch_size, channels), dtype=source.dtype))


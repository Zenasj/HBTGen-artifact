from tensorflow.keras import layers
from tensorflow.keras import models

import keras
import tensorflow as tf
import numpy as np


class UnAveragePooling2D(keras.layers.Layer):

    def __init__(self, kernel, *, strides, name = 'co1', dtype = None):
        super().__init__(trainable=True, name=name, dtype=dtype)
        self.kernel = kernel
        self.strides = strides

    def build(self, input_shape):
        pass

    def call(self, inputs):
        inputs_shape = tf.shape(inputs)

        # the averaged image had a minimum size of source.shape * strides, but it could have been larger if its size
        # was not a multiple of stride. For now, assume it was a multiple of stride.
        dest_shape = (inputs_shape[0], inputs.shape[1] * self.strides, inputs.shape[2] * self.strides, inputs_shape[3])
        dest = tf.zeros(dest_shape)

        # width of destination invalid edges that need to be filled, again assuming that source size was a multiple of stride.
        dest_start_invalid = (self.strides // 2, self.strides // 2)
        dest_end_invalid = (self.strides // 2, self.strides // 2)

        strides_f = tf.cast(self.strides, tf.float32)
        dest_start_invalid_f = (tf.cast(dest_start_invalid[0], tf.float32), tf.cast(dest_start_invalid[1], tf.float32))
        max_source_f = (tf.cast(inputs_shape[1] - 1, tf.float32), tf.cast(inputs_shape[2] - 1, tf.float32))

        # biniliear interpolation with distorted edges
        for dest_r in range(dest_shape[1]):
            for dest_c in range(dest_shape[2]):
                source_r = self._dest_to_rource(dest_r, strides_f, dest_start_invalid_f[0], max_source_f[0])
                source_c = self._dest_to_rource(dest_c, strides_f, dest_start_invalid_f[1], max_source_f[1])
                value = self._bilinear_interpolate(inputs, source_r, source_c)
                dest = self._assign_pixel_batch_values(dest, dest_r, dest_c, value)

        # # nearest edge fill
        # first_valid_row = tf.reshape(dest[:, dest_start_invalid[0], :, :], (dest_shape[0], 1, dest_shape[2], dest_shape[3]))
        # last_valid_row = tf.reshape(dest[:, -dest_end_invalid[0] - 1, :, :], (dest_shape[0], 1, dest_shape[2], dest_shape[3]))
        # start_rows = tf.tile(first_valid_row, (1, dest_start_invalid[0], 1, 1))
        # middle_rows = dest[:, dest_start_invalid[0]:-dest_end_invalid[0], :, :]
        # end_rows = tf.tile(last_valid_row, (1, dest_end_invalid[0], 1, 1))

        # dest = tf.concat([start_rows, middle_rows, end_rows], axis=1)
        # first_valid_col = tf.reshape(dest[:, :, dest_start_invalid[1], :], (dest_shape[0], dest_shape[1], 1, dest_shape[3]))
        # last_valid_col = tf.reshape(dest[:, :, -dest_end_invalid[1] - 1, :], (dest_shape[0], dest_shape[1], 1, dest_shape[3]))
        # start_cols = tf.tile(first_valid_col, (1, 1, dest_start_invalid[1], 1))
        # middle_cols = dest[:, :, dest_start_invalid[1]:-dest_end_invalid[1], :]
        # end_cols = tf.tile(last_valid_col, (1, 1, dest_end_invalid[1], 1))
        # dest = tf.concat([start_cols, middle_cols, end_cols], axis=2)

        return dest

    @tf.function
    def _dest_to_rource(self, dest, stride, dest_start_invalid, max_source):
        """Given a destination pixel row or column position, work out the source
        pixel location from which the value should be interpolated."""

        if dest < dest_start_invalid + stride - 0.5:
            return (dest - dest_start_invalid) / (stride - 0.5)
        elif dest > dest_start_invalid + (max_source - 1.0) * stride - 0.5:
            return ((dest - dest_start_invalid + 0.5) - (max_source - 1.0) * stride) / (stride - 0.5) + max_source - 1.0
        else:
            return (dest - dest_start_invalid + 0.5) / stride

    @tf.function
    def _bilinear_interpolate(self, source, r, c):
        """Given a batch of source images, interpolate each from the four pixels surrounding point r,c.
        Near the edge, fade to black.
        """
        # Algorithm (ignoring edges):
        # 1. Round r,c down to get top-right source pixel r0, c0
        # 2. Subtract r0, c0 from r, c get 0..1 proportions of a pixel fr, fc
        # 3. The four pixels to be sampled are at p00=[r0, c0], p01=[r0,c0+1], p10=[r0+1, c0], and p11=[r0+1, c0+1]
        # 4. For each channel, calculate a linear sum of the four pixels, as follows:
        # 5. result =   p00.(1-fr).(1-fc)
        #             + p01.(1-fr).fc
        #             + p10.fr.(1-fc)
        #             + p11.fr.fc
        # To cope with edges and implement fade-to-black: if any of p00, p01, p10 or p11 would be outside the source
        # image then use black instead
        r0, c0 =  tf.cast(tf.floor(r), tf.int32), tf.cast(tf.floor(c), tf.int32)
        fr, fc = r - tf.cast(r0, tf.float32), c - tf.cast(c0, tf.float32)
        p00 = self._safe_lookup_pixel(source, r0, c0)
        p01 = self._safe_lookup_pixel(source, r0, c0+1)
        p10 = self._safe_lookup_pixel(source, r0+1, c0)
        p11 = self._safe_lookup_pixel(source, r0+1, c0+1)
        return p00*(1-fr)*(1-fc) + p01*(1-fr)*fc + p10*fr*(1-fc) + p11*fr*fc

    @tf.function
    def _safe_lookup_pixel(self, source, r, c):
        """If x,y is a valid index into the source images then return the
        batch of pixels at that location, otherwise return a batch of black pixels."""

        source_shape = tf.shape(source)
        if 0 <= r < source_shape[1] and 0 <= c < source_shape[2]:
            return source[:, r, c, :]
        else:
            return tf.zeros((source_shape[0], source_shape[3]))

    @staticmethod
    @tf.function
    def _assign_pixel_batch_values(tensor, r, c, value):
        """Given a 4d tensor, replace the batch of pixels at location r,c with the given batch of pixels.
        Equivalent to tensor[:, r, c, :] = value
        """

        # There has got to be an easier way!
        tensor_shape = tensor.shape
        ret = tensor[:, :r, :, :]  # rows before the update
        new_row = tf.reshape(tensor[:, r, :c, :], (tensor_shape[0], 1, c, tensor_shape[3])) # columns before the update, on the update row
        value = tf.reshape(value, (tensor_shape[0], 1, 1, tensor_shape[3]))
        new_row = tf.concat([new_row, value], axis=2)   # append the new value
        last_cols = tf.reshape(tensor[:, r, c+1:, :], (tensor_shape[0], 1, tensor_shape[2] - c - 1, tensor_shape[3]))
        new_row = tf.concat([new_row, last_cols], axis=2) # columns after the update, on the update row
        ret = tf.concat([ret, new_row], axis=1)  # the now row with the update in it
        ret = tf.concat([ret, tensor[:, r+1:, :, :]], axis=1) # rows after the update
        return ret



TEST_SIZE = 16

data = [list(range(TEST_SIZE))] * TEST_SIZE
data = tf.convert_to_tensor(data, np.float32)
data = tf.reshape(data, (1, TEST_SIZE, TEST_SIZE, 1))

model = keras.models.Sequential()
model.add(keras.layers.AveragePooling2D((6, 6), padding='same', strides=4))
model.add(UnAveragePooling2D((6, 6), strides=4))
model.compile(run_eagerly=False)

expanded = model.predict(data)
print(expanded[0,:,:,0])

### Relevant log output
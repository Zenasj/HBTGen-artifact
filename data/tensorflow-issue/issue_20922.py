# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for slim.data.tfexample_decoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.slim.python.slim.data import tfexample_decoder
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test


class TFExampleDecoderTest(test.TestCase):

    def _EncodedFloatFeature(self, ndarray):
        return feature_pb2.Feature(float_list=feature_pb2.FloatList(
            value=ndarray.flatten().tolist()))

    def _EncodedInt64Feature(self, ndarray):
        return feature_pb2.Feature(int64_list=feature_pb2.Int64List(
            value=ndarray.flatten().tolist()))

    def _EncodedBytesFeature(self, tf_encoded):
        with self.test_session():
            encoded = tf_encoded.eval()

        def BytesList(value):
            return feature_pb2.BytesList(value=[value])

        return feature_pb2.Feature(bytes_list=BytesList(encoded))

    def _BytesFeature(self, ndarray):
        values = ndarray.flatten().tolist()
        for i in range(len(values)):
            values[i] = values[i].encode('utf-8')
        return feature_pb2.Feature(bytes_list=feature_pb2.BytesList(value=values))

    def _StringFeature(self, value):
        value = value.encode('utf-8')
        return feature_pb2.Feature(bytes_list=feature_pb2.BytesList(value=[value]))

    def _Encoder(self, image, image_format):
        assert image_format in ['jpeg', 'JPEG', 'png', 'PNG', 'raw', 'RAW']
        if image_format in ['jpeg', 'JPEG']:
            tf_image = constant_op.constant(image, dtype=dtypes.uint8)
            return image_ops.encode_jpeg(tf_image)
        if image_format in ['png', 'PNG']:
            tf_image = constant_op.constant(image, dtype=dtypes.uint8)
            return image_ops.encode_png(tf_image)
        if image_format in ['raw', 'RAW']:
            return constant_op.constant(image.tostring(), dtype=dtypes.string)

    def GenerateImage(self, image_format, image_shape):
        """Generates an image and an example containing the encoded image.

        Args:
          image_format: the encoding format of the image.
          image_shape: the shape of the image to generate.

        Returns:
          image: the generated image.
          example: a TF-example with a feature key 'image/encoded' set to the
            serialized image and a feature key 'image/format' set to the image
            encoding format ['jpeg', 'JPEG', 'png', 'PNG', 'raw'].
        """
        num_pixels = image_shape[0] * image_shape[1] * image_shape[2]
        image = np.linspace(
            0, num_pixels - 1, num=num_pixels).reshape(image_shape).astype(np.uint8)
        tf_encoded = self._Encoder(image, image_format)
        example = example_pb2.Example(features=feature_pb2.Features(feature={
            'image/encoded': self._EncodedBytesFeature(tf_encoded),
            'image/format': self._StringFeature(image_format)
        }))

        return image, example.SerializeToString()

    def DecodeExample(self, serialized_example, item_handler, image_format):
        """Decodes the given serialized example with the specified item handler.

        Args:
          serialized_example: a serialized TF example string.
          item_handler: the item handler used to decode the image.
          image_format: the image format being decoded.

        Returns:
          the decoded image found in the serialized Example.
        """
        serialized_example = array_ops.reshape(serialized_example, shape=[])
        decoder = tfexample_decoder.TFExampleDecoder(
            keys_to_features={
                'image/encoded':
                    parsing_ops.FixedLenFeature(
                        (), dtypes.string, default_value=''),
                'image/format':
                    parsing_ops.FixedLenFeature(
                        (), dtypes.string, default_value=image_format),
            },
            items_to_handlers={'image': item_handler})
        [tf_image] = decoder.decode(serialized_example, ['image'])
        return tf_image

    def RunDecodeExample(self, serialized_example, item_handler, image_format):
        tf_image = self.DecodeExample(serialized_example, item_handler,
                                      image_format)

        with self.test_session():
            decoded_image = tf_image.eval()

            # We need to recast them here to avoid some issues with uint8.
            return decoded_image.astype(np.float32)

    def testDecodeExampleWithPngEncodingAt16Bit(self):
        image_shape = (2, 3, 3)
        unused_image, serialized_example = self.GenerateImage(
            image_format='png', image_shape=image_shape)
        unused_decoded_image = self.RunDecodeExample(
            serialized_example,
            tfexample_decoder.Image(dtype=dtypes.uint16),
            image_format='png')
        self.assertAllClose(unused_image, unused_decoded_image)


if __name__ == '__main__':
    test.main()

def testDecodeExampleWithPngEncodingAt16Bit(self):
    image_shape = (2, 3, 3)
    unused_image, serialized_example = self.GenerateImage(
        image_format='png', image_shape=image_shape, dtype=np.uint16)
    unused_decoded_image = self.RunDecodeExample(
          serialized_example,
          tfexample_decoder.Image(dtype=dtypes.uint16),
          image_format='png')
    self.assertAllClose(unused_image, unused_decoded_image)
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.python.ops import gen_math_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import constant_op
import tensorflow as tf
import numpy as np

class RepeatLayers(tf.keras.layers.Layer):
    def __init__(self, axis=0):
        super(RepeatLayers, self).__init__()
        self.axis = axis

    def _all_dimensions(self, x):
        if isinstance(x, ops.Tensor) and x.get_shape().ndims is not None:
          return constant_op.constant(
              np.arange(x.get_shape().ndims), dtype=tf.int32)
        if (isinstance(x, sparse_tensor.SparseTensor) and x.dense_shape.get_shape().is_fully_defined()):
          r = x.dense_shape.get_shape().dims[0].value
          return constant_op.constant(tf.arange(r), dtype=tf.int32)

        return gen_math_ops._range(0, rank(x), 1)
    
    def _tile_one_dimension(self, data, axis, multiple):
        if data.shape.ndims is not None:
          multiples = [1] * data.shape.ndims
          multiples[axis] = multiple
        else:
          ones_value = tf.ones(tf.rank(data), tf.int32)
          multiples = tf.concat([ones_value[:axis], [multiple], ones_value[axis + 1:]],
                           axis=0)
      
        return tf.tile(data, multiples)

    def repeat_with_axis(self, data, repeats, axis):
        data = tf.convert_to_tensor(data, name='data') # [B, max_len, d]
        repeats = tf.cast(tf.convert_to_tensor(repeats, name='repeats'), tf.int32) # [B, max_len]

        data_shape = tf.shape(data)

        max_repeat = gen_math_ops.maximum(0, gen_math_ops._max(repeats, self._all_dimensions(repeats)))
        mask = tf.sequence_mask(repeats, max_repeat) # [B, max_len, max_value_of_repeat]

        expanded = tf.expand_dims(data, axis+1) # [B, max_len, 1, d]
        tiled = self._tile_one_dimension(expanded, axis+1, max_repeat) # [B, max_len, max_value_of_repeat, d]

        masked = tf.boolean_mask(tiled, mask) 
        result_shape = tf.concat([data_shape[:axis], [-1], data_shape[axis + 1:]], axis=0)
        result = tf.reshape(masked, result_shape)

        return result

    def call(self, encoder_h, repeats):
        return self.repeat_with_axis(data=encoder_h, repeats=repeats, axis=self.axis)


repeat = RepeatLayers(axis=1)

a = tf.keras.Input(shape=[35, 384], dtype=tf.float32)
b = tf.keras.Input(shape=[35], dtype=tf.int32)

output = repeat(a, b)

model = tf.keras.models.Model([a,b], outputs=output)


model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()



interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


input_shape_1 = input_details[0]['shape']
input_shape_2 = input_details[1]['shape']
input_data_1 = np.array(np.random.random_sample(input_shape_1), dtype=np.float32)
input_data_2 = np.array(np.random.random_sample(input_shape_2), dtype=np.int32)


interpreter.set_tensor(input_details[0]['index'], input_data_1)
interpreter.set_tensor(input_details[1]['index'], input_data_2)



interpreter.invoke()

interpreter.invoke()
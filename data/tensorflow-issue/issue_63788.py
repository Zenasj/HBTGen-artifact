from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import keras
import tensorflow as tf


class CustomCastLayer(keras.layers.Layer):
    def __init__(self, target_dtype, **kwargs):
        self.target_dtype = target_dtype
        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        Do Nothing
        """
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        output = tf.cast(inputs, self.target_dtype)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"target_dtype": self.target_dtype})
        return config

    def compute_output_shape(self, input_shape):
        """
        Do Nothing
        """
        return input_shape


class CustomPadLayer(keras.layers.Layer):
    def __init__(self, padding=[[0, 0]], constant_values=0, **kwargs):
        self.padding = padding  # add [0,0] to padding so we will not change the shape of batch dimension
        self.constant_values = constant_values
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        output = tf.pad(inputs, paddings=tf.constant([[0, 0]] + self.padding), mode="CONSTANT",
                        constant_values=self.constant_values)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"padding": self.padding, "constant_values": self.constant_values})
        return config

    def compute_output_shape(self, input_shape):
        """
        Formula to calculate the output shape
        Suppose the input_shape is (None, N, W, C), the `paddings` is [[a, b], [c, d]].
        The output_shape is (None, N+a+b, W+c+d, C).
        """
        input_shape_list = list(input_shape)
        padding = [[0, 0]] + self.padding
        assert len(input_shape_list) == len(padding)  # Two dimensions should match.
        output_shape = [None]
        for i, pad in zip(input_shape_list[1:], padding[1:]):
            output_shape.append(i + pad[0] + pad[1])
        return tuple(output_shape)


class CustomCropLayer(keras.layers.Layer):
    def __init__(self, cropping, **kwargs):
        self.cropping = cropping
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        input_shape = inputs.shape.as_list()
        indices = [slice(None)]
        cropping = [[0, 0]] + self.cropping  # add [0,0] to padding so we will not change the shape of batch dimension
        for shape, crop in zip(input_shape[1:], cropping[1:]):
            indices.append(slice(0 + crop[0], shape - crop[1]))
        return inputs[indices]

    def get_config(self):
        config = super().get_config()
        config.update({"cropping": self.cropping})
        return config

    def compute_output_shape(self, input_shape):
        """
        Formula to calculate the output shape
        Suppose the input_shape is (None, N, W, C), the `cropping` is [[a, b], [c, d]].
        The output_shape is (None, N-a-b, W-c-d, C).
        """
        input_shape_list = list(input_shape)
        cropping = [[0, 0]] + self.cropping
        assert len(input_shape_list) == len(cropping)  # Two dimensions should match.
        output_shape = [None]
        for i, crop in zip(input_shape[1:], cropping[1:]):
            output_shape.append(i - crop[0] - crop[1])
        return tuple(output_shape)


class CustomExpandLayer(keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.expand_dims(inputs, self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

    def compute_output_shape(self, input_shape):
        """
        Formula to calculate the output shape
        Suppose the input_shape is [None, N, W, C]:
        axis=0:
            output_shape: [1, None, N, W, C]
        axis=1 (default):
            output_shape: [None, 1, N, W, C]
        axis=2:
            output_shape: [None, N, 1, W, C]
        axis=3:
            output_shape: [None, N, W, 1, C]
        axis=4:
            output_shape: [None, N, W, C, 1]
        axis=5:
            raise Exception
        """
        input_shape_list = list(input_shape)
        if self.axis > len(input_shape_list):
            raise ValueError(f"axis {self.axis} should be smaller than input_shape + 1: {len(input_shape_list) + 1}")
        output_shape = input_shape_list[0:self.axis] + [1] + input_shape_list[self.axis:]
        return tuple(output_shape)  # we should use tuple!!! not list !!!


class CustomDropDimLayer(keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Something magic to automatically generate indices for array slicing.
        To determine a specific axis, we can use slice(None) to replace `:`
        """
        dim = len(inputs.shape)
        if self.axis > dim - 1 or self.axis < 1:
            raise ValueError(f"axis: {self.axis} should be within the range: [1, {dim - 1}] for {dim}D tensor")
        indices = [slice(None) for i in range(dim)]
        indices[self.axis] = 0
        return inputs[indices]

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

    def compute_output_shape(self, input_shape):
        """
        Formula to calculate the output shape
        Suppose the input_shape is [None, N, W, C]:
        axis=0:  # Although it is feasible, we don't allow this to happen
            Raise Exception
        axis=1 (default):
            output_shape: [None, W, C]
        axis=2:
            output_shape: [None, N, C]
        axis=3:
            output_shape: [None, N, W]
        axis=4:
            Raise Exception
        """
        input_shape_list = list(input_shape)
        output_shape = input_shape_list[0:self.axis] + input_shape_list[self.axis + 1:]
        return tuple(output_shape)


def custom_objects(mode="custom"):
    def no_activation(x):
        return x

    def leakyrelu(x):
        import keras.backend as K
        return K.relu(x, alpha=0.01)

    # objects = {}
    objects = {'no_activation': no_activation, 'leakyrelu': leakyrelu}
    if mode == "custom":
        objects['CustomPadLayer'] = CustomPadLayer
        objects['CustomCropLayer'] = CustomCropLayer
        objects['CustomDropDimLayer'] = CustomDropDimLayer
        objects['CustomExpandLayer'] = CustomExpandLayer
        objects['CustomCastLayer'] = CustomCastLayer

    return objects


model_path = "/data1/pzy/MUTANTS/LEMON/lenet5_fashion/lenet5-fashion-mnist_origin-NLAll30-LMerg38-Edge63-NLAll90-92/lenet5-fashion-mnist_origin-NLAll30-LMerg38-Edge63-NLAll90.h5"
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects())
# model = tf.keras.models.load_model(model_path,)
model.save(f'tmp/lenet5_sb', save_format='tf')
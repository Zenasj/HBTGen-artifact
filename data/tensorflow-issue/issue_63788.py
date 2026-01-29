# tf.random.uniform((None, None, None, 32), dtype=tf.float32)
import tensorflow as tf
from tensorflow import keras

class CustomCastLayer(keras.layers.Layer):
    def __init__(self, target_dtype, **kwargs):
        self.target_dtype = target_dtype
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        output = tf.cast(inputs, self.target_dtype)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"target_dtype": self.target_dtype})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class CustomPadLayer(keras.layers.Layer):
    def __init__(self, padding=[[0, 0]], constant_values=0, **kwargs):
        self.padding = padding  # add [0,0] for batch dimension to not change
        self.constant_values = constant_values
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # paddings need to include batch dim as no padding
        paddings = tf.constant([[0, 0]] + self.padding)
        output = tf.pad(inputs, paddings=paddings, mode="CONSTANT",
                        constant_values=self.constant_values)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"padding": self.padding, "constant_values": self.constant_values})
        return config

    def compute_output_shape(self, input_shape):
        """
        Suppose input_shape = (None, N, W, C), padding=[[a,b],[c,d]]
        output_shape = (None, N+a+b, W+c+d, C)
        """
        input_shape_list = list(input_shape)
        padding = [[0, 0]] + self.padding
        assert len(input_shape_list) == len(padding)
        output_shape = [None]
        for i, pad in zip(input_shape_list[1:], padding[1:]):
            if i is None:
                output_shape.append(None)
            else:
                output_shape.append(i + pad[0] + pad[1])
        return tuple(output_shape)


class CustomCropLayer(keras.layers.Layer):
    def __init__(self, cropping, **kwargs):
        self.cropping = cropping
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # Here inputs.shape may have None for dynamic dims.
        # The original error occurs because shape[i] was None and 
        # subtracting crop[1] fails.
        # To fix this, use tf.shape(inputs) for runtime shape (all ints),
        # instead of inputs.shape.as_list() which can have None.

        input_shape_dynamic = tf.shape(inputs)
        # cropping includes batch dim no crop
        cropping = [[0, 0]] + self.cropping
        indices = [slice(None)]  # batch dim slice all

        # Build slices for spatial dims safely
        # Use tf.slice or tf.strided_slice by building begin/end tensors

        # We'll convert slices to tf.strided_slice calls processing axes 1 to N
        slices = [slice(None)]  # batch dim

        # For each dim after batch dim, crop appropriately
        begin = [0]
        size = [-1]  # dynamic length
        for i in range(1, len(cropping)):
            crop_start, crop_end = cropping[i]
            dim_len = input_shape_dynamic[i]
            # Calculate start and length for slice with crop
            start = crop_start
            # For size, length = dim_len - crop_start - crop_end
            length = dim_len - crop_start - crop_end
            begin.append(start)
            size.append(length)

        # Use tf.slice for cropping: slice(inputs, begin, size)
        output = tf.slice(inputs, begin, size)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"cropping": self.cropping})
        return config

    def compute_output_shape(self, input_shape):
        """
        Suppose input_shape = (None, N, W, C), cropping=[[a,b],[c,d]]
        output_shape = (None, N - a - b, W - c - d, C)
        """
        input_shape_list = list(input_shape)
        cropping = [[0, 0]] + self.cropping
        assert len(input_shape_list) == len(cropping)
        output_shape = [None]
        for i, crop in zip(input_shape_list[1:], cropping[1:]):
            if i is None:
                output_shape.append(None)
            else:
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
        input_shape_list = list(input_shape)
        if self.axis > len(input_shape_list):
            raise ValueError(f"axis {self.axis} should be smaller than input_shape+1: {len(input_shape_list) + 1}")
        output_shape = input_shape_list[0:self.axis] + [1] + input_shape_list[self.axis:]
        return tuple(output_shape)


class CustomDropDimLayer(keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # Restrict axis to be in range [1, dim-1] for inputs
        dim = len(inputs.shape)
        if self.axis > dim - 1 or self.axis < 1:
            raise ValueError(f"axis: {self.axis} should be within [1, {dim-1}] for {dim}D tensor")
        indices = [slice(None) for _ in range(dim)]
        indices[self.axis] = 0
        return inputs[tuple(indices)]

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

    def compute_output_shape(self, input_shape):
        input_shape_list = list(input_shape)
        output_shape = input_shape_list[0:self.axis] + input_shape_list[self.axis + 1:]
        return tuple(output_shape)


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # For demonstration, the model will just apply a chain of 
        # CustomCropLayer, CustomPadLayer, CustomCastLayer, CustomExpandLayer, CustomDropDimLayer
        # in sequence to show usage of layers and safe input handling.
        # This is a minimal reproducible setup mimicking the user code context.
        self.crop = CustomCropLayer(cropping=[[2, 2], [2, 2], [0, 0]])
        self.pad = CustomPadLayer(padding=[[1, 1], [1, 1], [0, 0]])
        self.cast = CustomCastLayer(target_dtype=tf.float32)
        self.expand = CustomExpandLayer(axis=1)
        self.dropdim = CustomDropDimLayer(axis=2)

    def call(self, inputs, training=False):
        # pipeline of these custom layers
        x = self.cast(inputs)
        x = self.pad(x)
        x = self.crop(x)
        x = self.expand(x)
        x = self.dropdim(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # The original error occurs with input of shape (None, None, None, 32)
    # So to test, we create a random tensor with shape (batch=1, height=28, width=28, channels=32) float32

    # The shape must be compatible with cropping/padding done in model:
    # cropping cuts 2 pixels each side on dims 1 and 2, padding adds 1 pixel each side, net shrinking by 2-1=1 per side -> output spatial dims smaller by 2
    # So input spatial dims must be at least 6 to avoid negative sizes.
    # Pick 28x28 for safe input size.

    B = 1
    H = 28
    W = 28
    C = 32
    return tf.random.uniform((B, H, W, C), dtype=tf.float32)


# tf.random.uniform((10, 28, 28, 1), dtype=tf.float32)  # Assumed input shape based on typical LeNet5 for Fashion-MNIST

import tensorflow as tf
from tensorflow.keras import layers

class CustomCastLayer(tf.keras.layers.Layer):
    def __init__(self, target_dtype, **kwargs):
        super().__init__(**kwargs)
        self.target_dtype = target_dtype

    def call(self, inputs):
        return tf.cast(inputs, self.target_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"target_dtype": self.target_dtype})
        return config

class CustomPadLayer(tf.keras.layers.Layer):
    def __init__(self, padding=[[0, 0]], constant_values=0, **kwargs):
        super().__init__(**kwargs)
        self.padding = padding
        self.constant_values = constant_values

    def call(self, inputs):
        paddings = tf.constant([[0, 0]] + self.padding)
        return tf.pad(inputs, paddings=paddings, mode="CONSTANT", constant_values=self.constant_values)

    def get_config(self):
        config = super().get_config()
        config.update({"padding": self.padding, "constant_values": self.constant_values})
        return config

class CustomCropLayer(tf.keras.layers.Layer):
    def __init__(self, cropping, **kwargs):
        super().__init__(**kwargs)
        self.cropping = cropping

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        slicing = [slice(None)]
        # cropping is [[top_crop, bottom_crop], [left_crop, right_crop]]
        cropping_full = [[0, 0]] + self.cropping  # keep batch dim unchanged
        for i, crop in enumerate(cropping_full[1:], start=1):
            slicing.append(slice(crop[0], input_shape[i] - crop[1]))
        return inputs[slicing]

    def get_config(self):
        config = super().get_config()
        config.update({"cropping": self.cropping})
        return config

class CustomExpandLayer(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

class CustomDropDimLayer(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        # Slice out the dimension at axis by selecting index 0 along that axis
        indices = [slice(None)] * len(inputs.shape)
        indices[self.axis] = 0
        return inputs[tuple(indices)]

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model reproduces the core Lenet5 + custom ops logic hinted in the issue,
        # focusing on the conv2d layer that unexpectedly outputs no NaNs given NaN inputs.

        # Assumptions: Input shape is (None, 28, 28, 1) typical for Fashion-MNIST grayscale

        # Basic Layers roughly reflecting Lenet5 architecture
        self.conv1 = layers.Conv2D(filters=6, kernel_size=5, activation='relu', padding='valid')
        self.pool1 = layers.MaxPooling2D(pool_size=2, strides=2)
        self.conv2 = layers.Conv2D(filters=16, kernel_size=5, activation='relu', padding='valid')
        self.pool2 = layers.MaxPooling2D(pool_size=2, strides=2)

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(120, activation='relu')
        self.dense2 = layers.Dense(84, activation='relu')
        self.dense3 = layers.Dense(10)  # 10 classes for Fashion-MNIST

        # Custom layers from the issue, used around dropout and cast operations
        self.custom_cast = CustomCastLayer(target_dtype=tf.float32)
        self.custom_pad = CustomPadLayer(padding=[[2, 2], [2, 2]], constant_values=0)
        self.custom_crop = CustomCropLayer(cropping=[[2, 2], [2, 2]])
        self.custom_expand = CustomExpandLayer(axis=1)
        self.custom_dropdim = CustomDropDimLayer(axis=1)

        # To simulate the logic where NaN input is expected to propagate through conv2d:
        # We add a lambda layer that can produce NaNs (like a dropout or similar)
        # However, dropout won't create NaNs, so we simulate a "nan induction" using tf.where-like.
        # But here we just handle as is.

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # The model forward path with NaN input test:
        x = inputs

        # First conv block
        x = self.conv1(x)
        x = self.pool1(x)

        # Introduce NaN manually if not present (for demonstration),
        # but as input can have NaNs, we trust input to have NaNs if testing is done.
        # Note: the reported issue is that conv2d outputs no NaNs despite NaN input.

        # Second conv block
        x = self.conv2(x)
        x = self.pool2(x)

        # Flatten and dense layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.dense3(x)

        # Output: return the final output + a boolean mask indicating if any NaNs exist in output
        nan_mask = tf.math.reduce_any(tf.math.is_nan(out))
        # Return both for inspection purposes (nan_mask is scalar bool)
        return out, nan_mask

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching expected input shape (batch=10, 28x28 grayscale image)
    # We also include NaNs to test the issue from the original problem.
    shape = (10, 28, 28, 1)
    inp = tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)

    # Introduce NaNs at random positions intentionally
    import numpy as np
    np_inp = inp.numpy()
    # Set about 5% of elements to NaN to simulate NaN inputs
    nan_mask = np.random.rand(*shape) < 0.05
    np_inp[nan_mask] = np.nan
    inp_with_nan = tf.convert_to_tensor(np_inp, dtype=tf.float32)
    return inp_with_nan


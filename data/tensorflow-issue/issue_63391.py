# tf.random.uniform((10, 28, 28, 1), dtype=tf.float32) ← inferred input shape based on typical LeNet5 Fashion MNIST input

import tensorflow as tf
from tensorflow.keras import layers

# Re-implementing the custom layers seen in the issue for compatibility and standalone use.
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
        # padding assumed on spatial dims only: height, width etc.
        paddings = [[0, 0]] + self.padding  # batch dim no pad
        paddings = tf.constant(paddings, dtype=tf.int32)
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
        # cropping is list of [ [top_crop, bottom_crop], [left_crop, right_crop], ...] for dims after batch
        input_shape = tf.shape(inputs)
        slices = [slice(None)]  # batch
        # for each spatial dimension:
        for i, (crop_start, crop_end) in enumerate(self.cropping, start=1):
            length = input_shape[i]
            slices.append(slice(crop_start, length - crop_end))
        return inputs[tuple(slices)]

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
        # This removes the specified dimension by indexing 0 at that axis
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        # Construct slicing indices: select index 0 at `axis`, slice(None) elsewhere
        dim = len(inputs.shape)
        if self.axis < 0:
            self.axis = dim + self.axis
        slices = [slice(None)] * dim
        slices[self.axis] = 0
        return inputs[tuple(slices)]

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


class MyModel(tf.keras.Model):
    """
    Reconstructing a simplified version of the LeNet5 variant model pipeline described,
    focusing on the core behavior around conv2d with NaN inputs and outputs,
    and the use of lambda and dropout layers which influence NaNs.

    We will implement the network with:
    - Input layer expecting (None, 28, 28, 1) typical for Fashion MNIST
    - A sequence of conv2d, maxpool, lambda, dropout, another conv2d, maxpool, dropout layers as per named info
    - We include the lambda and dropout layers as pass-through but demonstrating NaN generation handling
    """

    def __init__(self):
        super().__init__()

        # Layer 0: Conv2D - mimic conv2d_1_copy_SpecialI_copy_LMerg
        self.conv2d_1 = layers.Conv2D(filters=6, kernel_size=5, activation='relu', padding='same', name='conv2d_1')

        # Layer 1: MaxPooling - mimic max_pooling2d_1_copy_SpecialI_copy_LMerg
        self.maxpool1 = layers.MaxPooling2D(pool_size=2, strides=2, name='max_pooling2d_1')

        # Layer 2: Lambda layer - acts as identity but can set NaNs for test
        # We'll implement a lambda layer that propagates NaNs (identity here)
        self.lambda_layer = layers.Lambda(lambda x: tf.where(tf.math.is_nan(x), tf.fill(tf.shape(x), float('nan')), x),
                                          name='lambda_copy_LMerg')

        # Layer 3: Dropout (training=True to induce NaNs with some probability during call)
        self.dropout1 = layers.Dropout(rate=0.5, name='dropout_1_copy_SpecialI_copy_LMerg')

        # Layer 4: Conv2D 2nd - mimic conv2d_2_copy_SpecialI_copy_LMerg
        self.conv2d_2 = layers.Conv2D(filters=16, kernel_size=5, activation='relu', padding='valid', name='conv2d_2')

        # Layer 5: MaxPooling 2nd
        self.maxpool2 = layers.MaxPooling2D(pool_size=2, strides=2, name='max_pooling2d_2')

        # Layer 6: Dropout 2nd
        self.dropout2 = layers.Dropout(rate=0.5, name='dropout_2_copy_SpecialI_copy_LMerg_merge1')

        # Flatten + dense layers to mimic remainder of structure
        self.flatten = layers.Flatten(name='flatten_1_copy_SpecialI_copy_LMerg')
        self.dense1 = layers.Dense(120, activation='relu', name='dense_1_copy_SpecialI_copy_LMerg')
        self.dense2 = layers.Dense(84, activation='relu', name='dense_2_copy_SpecialI_copy_LMerg')
        self.dense3 = layers.Dense(10, activation='softmax', name='dense_3_copy_SpecialI_copy_LMerg')

        # Extra dense layer mimicking 'dense_insert' layer from output list
        self.dense_insert = layers.Dense(10, activation='linear', name='dense_insert')

    def call(self, inputs, training=None):
        # Propagate inputs through the layers
        x = self.conv2d_1(inputs)       # Layer 1 conv (expected NaN propagation here if input has NaN)
        x = self.maxpool1(x)             # Layer 2 maxpool

        x = self.lambda_layer(x)         # Layer 3 lambda (identity but preserves NaNs)

        # Dropout only applies in training mode, will propagate NaNs in dropped units
        x = self.dropout1(x, training=training)  # Layer 4 dropout

        x = self.conv2d_2(x)             # Layer 5 conv2d

        x = self.maxpool2(x)             # Layer 6 maxpool

        x = self.dropout2(x, training=training) # Layer 7 dropout

        x = self.flatten(x)              # Layer 8 flatten

        x = self.dense1(x)               # Layer 9 dense1
        x = self.dense2(x)               # Layer 10 dense2
        x = self.dense3(x)               # Layer 11 dense3

        x = self.dense_insert(x)         # Layer 12 dense insert

        return x


def my_model_function():
    """
    Return an instance of the reconstructed MyModel.
    No pretrained weights loaded here since the h5 file and custom lambda layers cannot be loaded reliably
    without original source.
    """
    return MyModel()


def GetInput():
    """
    Return a batch of input tensors:
    - Batch size 10 (as in original code replacing batch dim with 10)
    - Shape (28, 28, 1) matching Fashion MNIST input format
    - Random floats in [0,1), with injected NaNs for demonstration below.

    We inject NaNs randomly in the input tensor to mirror user interest in NaN propagation.
    """

    import numpy as np

    batch_size = 10
    height = 28
    width = 28
    channels = 1

    np.random.seed(42)
    input_np = np.random.rand(batch_size, height, width, channels).astype(np.float32)

    # Inject NaNs randomly in 10% of the elements
    nan_mask = np.random.rand(*input_np.shape) < 0.1
    input_np[nan_mask] = np.nan

    input_tensor = tf.convert_to_tensor(input_np)

    return input_tensor


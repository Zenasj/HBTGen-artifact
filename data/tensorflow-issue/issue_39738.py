# tf.random.uniform((B, 16, 16, 3), dtype=tf.float32)  ← Input shape inferred from the example code's inputs shape [5,16,16,3]

import tensorflow as tf

class CustomSequential(tf.keras.Sequential):
    def __init__(self, set_attributes=False, name='CustomSequential', **kwargs):
        # The sequential model contains 2 conv layers, flatten, dense as per original code
        layers = [
            tf.keras.layers.Conv2D(2, 3, 2, name='conv2'),
            tf.keras.layers.Conv2D(4, 3, 2, name='conv3'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10)
        ]
        super().__init__(layers, name=name, **kwargs)

        # Workaround: manually set layer attributes to enable checkpoint tracking in TF 2.2+
        # This corresponds to the "set_attributes" flag controlling attribute assignment.
        if set_attributes:
            self.conv1 = self.layers[0]  # conv2 layer
            self.conv2 = self.layers[1]  # conv3 layer
            self.dense = self.layers[3]


class CustomModel(tf.keras.Model):
    def __init__(self, set_attributes, name='CustomModel', **kwargs):
        super().__init__(name=name, **kwargs)
        # First convolution layer before the sequential block
        self.convolution = tf.keras.layers.Conv2D(8, 3, name='conv1')
        # CustomSequential block with possible attribute assignment
        self.sequential = CustomSequential(set_attributes, name='MySequential')

    def call(self, inputs):
        net = self.convolution(inputs)
        return self.sequential(net)


# Following task instructions: fuse models or comparison logic if multiple models exist.
# Here we have only one model architecture (CustomModel with CustomSequential inside).
# We'll encapsulate the provided CustomModel behavior into MyModel.

class MyModel(tf.keras.Model):
    def __init__(self, set_attributes=True, name='MyModel', **kwargs):
        super().__init__(name=name, **kwargs)
        # Instantiate the internal CustomModel logic
        self.convolution = tf.keras.layers.Conv2D(8, 3, name='conv1')
        self.sequential = CustomSequential(set_attributes, name='MySequential')

    def call(self, inputs):
        x = self.convolution(inputs)
        return self.sequential(x)


def my_model_function():
    # Return instance of MyModel
    # Default to set_attributes=True to follow the recommended workaround to track layers
    return MyModel(set_attributes=True)


def GetInput():
    # As per original example, input shape is [B, 16, 16, 3] with dtype float32
    # Let's generate a random tensor with batch size 5 (as in example)
    B = 5
    H = 16
    W = 16
    C = 3
    return tf.random.uniform((B, H, W, C), dtype=tf.float32)


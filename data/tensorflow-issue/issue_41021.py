# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← Input shape inferred from CIFAR-10 dataset used in the issue

import tensorflow as tf
from tensorflow.keras import layers

class NestedLayers(tf.keras.layers.Layer):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define three Conv2D layers each with unique names
        # Only first two Conv2D layers are used in build and call, consistent with original code
        self.units = [
            layers.Conv2D(16, (3, 3), name="conv_2d_0"),
            layers.Conv2D(16, (3, 3), name="conv_2d_1"),
            layers.Conv2D(16, (3, 3), name="conv_2d_2"),
        ]

    def build(self, input_shape):
        # Build only first two conv layers with modified input shape (channels set to 1)
        for i in range(0, 2):
            unit_input_shape = list(input_shape)
            unit_input_shape[-1] = 1
            unit = self.units[i]
            # Use a unique name scope during build to avoid naming collisions in weights when saving
            with tf.name_scope(f"BUILD_{i}"):
                unit.build(unit_input_shape)
        super(NestedLayers, self).build(input_shape)  # call parent build

    def call(self, inputs):
        # Split inputs into 3 groups along channels axis, use only first two groups for conv units
        split_inputs = tf.split(value=inputs, num_or_size_splits=3, axis=-1, name="conv_grp_split")
        outputs = []
        for i in range(0, 2):
            out = self.units[i](split_inputs[i])
            outputs.append(out)
        # Concatenate outputs of first two conv layers along channels
        out = tf.keras.layers.concatenate(outputs, axis=-1, name="conv_grp_concat")
        return out

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Input shape is expected to be (32,32,3) float32 batches
        self.nested = NestedLayers()
        self.flatten = layers.Flatten()
        # Final dense layer for 10-class classification (CIFAR-10)
        self.classifier = layers.Dense(10, activation='softmax', name="classifier")

    def call(self, inputs):
        x = self.nested(inputs)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

def my_model_function():
    # Instantiates the MyModel class 
    model = MyModel()
    # Build the model by calling it once on dummy data to create weights
    dummy_input = GetInput()
    _ = model(dummy_input)
    return model

def GetInput():
    # Return a batch of random float32 tensors shaped (B, 32, 32, 3)
    # Batch size arbitrarily chosen as 4 here
    batch_size = 4
    return tf.random.uniform((batch_size, 32, 32, 3), dtype=tf.float32)


# tf.random.uniform((BATCH_SIZE, 300, 300, 3), dtype=tf.float32) ← input shape inferred from ISSUE describing a 300x300 RGB input image

import tensorflow as tf

# Constants (inferred from issue context)
IMAGE_SIZE = (300, 300)
CLASSES = 10  # Assumed number of classes, as 'CLASSES' was undefined in issue; adjust as needed.

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=CLASSES):
        super().__init__()
        # Initialize two EfficientNets with imagenet weights, excluding top
        # Rename their layers and weights to avoid naming conflicts when combining
        self.efficient_net0 = tf.keras.applications.EfficientNetB0(
            weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3), name='efficientnetb0_custom'
        )
        self.rename_layers_and_weights(self.efficient_net0, suffix="_0")

        self.efficient_net1 = tf.keras.applications.EfficientNetB1(
            weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3), name='efficientnetb1_custom'
        )
        self.rename_layers_and_weights(self.efficient_net1, suffix="_1")

        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()

        # Separate Dense layers for each EfficientNet output with unique names
        self.dense0 = tf.keras.layers.Dense(num_classes, activation="sigmoid", dtype='float32', name='dense_0')
        self.dense1 = tf.keras.layers.Dense(num_classes, activation="sigmoid", dtype='float32', name='dense_1')

        # Final output averaging layer (implemented in call)

    def rename_layers_and_weights(self, model, suffix):
        # Rename layers by appending suffix to avoid name collisions
        for layer in model.layers:
            # Append suffix if not already present
            if not layer.name.endswith(suffix):
                layer._name = layer.name + suffix

        # Rename the underlying weights similarly (hacky but works)
        # This is the critical fix discussed in the issue to avoid "name already exists" error on saving
        for w in model.weights:
            if not w.name.endswith(suffix + ':0'):
                # Modify the private _handle_name attribute to give weights unique names
                # We use '_handle_name' because it controls the HDF5 internal names used by the save process
                new_handle_name = 'custom_' + suffix.strip('_') + '_' + w.name
                w._handle_name = new_handle_name

    def call(self, inputs, training=False):
        # Pass input through EfficientNet 0
        x0 = self.efficient_net0(inputs, training=training)
        x0 = self.global_avg_pool(x0)
        out0 = self.dense0(x0)

        # Pass input through EfficientNet 1
        x1 = self.efficient_net1(inputs, training=training)
        x1 = self.global_avg_pool(x1)
        out1 = self.dense1(x1)

        # Average outputs as final prediction (as per original code using tf.keras.layers.average)
        out = (out0 + out1) / 2.0
        return out


def my_model_function():
    """
    Returns an instance of MyModel.
    This includes the renamed layers and weights to avoid name collisions.
    """
    return MyModel()


def GetInput():
    """
    Returns a random input tensor matching the model input expectations:
    Shape: (batch_size, 300, 300, 3)
    Data type: tf.float32
    Values uniform random [0,1)
    """
    BATCH_SIZE = 8  # Arbitrary batch size for testing; adjust as needed
    return tf.random.uniform(shape=(BATCH_SIZE, *IMAGE_SIZE, 3), dtype=tf.float32)


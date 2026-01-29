# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← inferred from ResNet50 Conv2D input (typical ImageNet size 224x224 with 3 channels)
import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We create two submodels analogous to the discussions about inconsistencies/errors with protobufs
        # and tensorflow.keras.models.Sequential usage. The fusion here is conceptual: two simple models to compare.
        # This example fuses a small CNN and a small MLP, simulating two possible models users might want to check
        # for equality or differences due to protobuf/tf changes.
        
        # CNN model: resembles a minimal Conv2D->Flatten->Dense pipeline similar to ResNet initial layers
        self.cnn = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False, name='conv1_conv'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu')
        ])

        # MLP model: simple flatten -> dense with linear activation
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='linear')
        ])

    def call(self, x, training=False):
        # Run both submodels on the input
        out_cnn = self.cnn(x, training=training)
        out_mlp = self.mlp(x, training=training)

        # Comparison logic:
        # Compute absolute difference, then a boolean indicating if all elements are close within a tolerance
        diff = tf.abs(out_cnn - out_mlp)
        # Use a tolerance inspired by typical floating point comparison (1e-5)
        tolerance = 1e-5
        are_close = tf.reduce_all(diff < tolerance)

        # Return a dictionary of outputs for introspection: cnn output, mlp output, difference magnitude, are_close boolean
        # This mirrors a scenario where one might want to debug differences caused by protobuf or tensorflow issues.
        return {
            'cnn_output': out_cnn,
            'mlp_output': out_mlp,
            'difference': diff,
            'are_close': are_close
        }


def my_model_function():
    # Return an instance of MyModel.
    # No pretrained weights or checkpoints are loaded as none were specified.
    return MyModel()


def GetInput():
    # Return a random uniform tensor matching the expected input of the model's CNN branch:
    # Typically a batch of RGB images at 224x224 resolution (standard ImageNet input for ResNet50)
    batch_size = 4
    height = 224
    width = 224
    channels = 3
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)


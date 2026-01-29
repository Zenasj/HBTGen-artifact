# tf.random.uniform((None, 100, 3), dtype=tf.float32) ← Inferred input shape from the original model InputLayer shape

import tensorflow as tf

# Placeholder for the FeatureSteeredConvolutionKerasLayer since it's from tensorflow_graphics
# which caused saving issues related to serialization. We'll mock it with simple Conv1D to maintain shape and flow.
class FeatureSteeredConvolutionKerasLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=1, **kwargs):
        super().__init__(**kwargs)
        # Using Conv1D as a placeholder for the graph convolution layer
        self.conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same')

    def call(self, inputs):
        # inputs is a tuple/list: [feature_tensor, adjacency_tensor]
        x = inputs[0]  # feature tensor
        # adjacency matrix ignored in placeholder
        return self.conv(x)


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # From model summary inferred layers:
        # Inputs: tensor shape (None, 100, 3) - point features
        #           (None, 100, 100) - adjacency or connectivity matrix
        #           (None,) - scalar or extra input (unused in model summary)
        
        # Initial Conv1D layer 3->2 channels
        self.conv1 = tf.keras.layers.Conv1D(2, kernel_size=1, padding='same')

        # Three Feature Steered Conv layers with increasing filters: 32, 64, 128
        self.fsc1 = FeatureSteeredConvolutionKerasLayer(32)
        self.relu1 = tf.keras.layers.ReLU()

        self.fsc2 = FeatureSteeredConvolutionKerasLayer(64)
        self.relu2 = tf.keras.layers.ReLU()

        self.fsc3 = FeatureSteeredConvolutionKerasLayer(128)
        self.relu3 = tf.keras.layers.ReLU()

        # Two final Conv1D layers: 128->256, then 256->1
        self.conv2 = tf.keras.layers.Conv1D(256, kernel_size=1, padding='same')
        self.conv3 = tf.keras.layers.Conv1D(1, kernel_size=1, padding='same')

    def call(self, inputs):
        # Unpack inputs tuple
        # Assumed shapes:
        # inputs[0] = features: (batch, 100, 3)
        # inputs[1] = adjacency: (batch, 100, 100)
        # inputs[2] = extra scalar inputs: (batch,) - not used in model layers per summary

        features, adjacency, extra_scalar = inputs

        x = self.conv1(features)  # (batch, 100, 2)

        x = self.fsc1([x, adjacency])  # (batch, 100, 32)
        x = self.relu1(x)

        x = self.fsc2([x, adjacency])  # (batch, 100, 64)
        x = self.relu2(x)

        x = self.fsc3([x, adjacency])  # (batch, 100, 128)
        x = self.relu3(x)

        x = self.conv2(x)  # (batch, 100, 256)

        x = self.conv3(x)  # (batch, 100, 1)

        # Return final output (e.g., per-point score or prediction)
        return x


def my_model_function():
    # Return an instance of the model
    return MyModel()


def GetInput():
    # Return a tuple of three inputs matching the expected model inputs:
    # shape (batch, 100, 3), (batch, 100, 100), (batch,)
    B = 4  # batch size arbitrary for demonstration

    # Features: float32 tensor (B, 100, 3)
    features = tf.random.uniform((B, 100, 3), dtype=tf.float32)

    # Adjacency matrices: float32 tensor (B, 100, 100), values in [0,1]
    adjacency = tf.random.uniform((B, 100, 100), dtype=tf.float32)

    # Extra input scalar per batch element, unused in layers, shape (B,)
    extra_scalar = tf.random.uniform((B,), dtype=tf.float32)

    return (features, adjacency, extra_scalar)


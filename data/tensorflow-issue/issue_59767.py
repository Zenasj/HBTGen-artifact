# tf.random.uniform((B, 572, 572, 3), dtype=tf.float32)  # Assuming input size ~572x572 RGB images as per issue context

import tensorflow as tf
from tensorflow.keras import layers, initializers, models

# Reimplementation of the missing ConvNeXt custom layers (LayerScale and StochasticDepth)
# based on the Keras ConvNeXt code references and context from the issue.

class LayerScale(layers.Layer):
    def __init__(self, projection_dim, init_values=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.init_values = init_values

    def build(self, input_shape):
        # Gamma is initialized as a scalar per channel with very small initial values
        self.gamma = self.add_weight(
            shape=(self.projection_dim,),
            initializer=initializers.Constant(self.init_values),
            trainable=True,
            dtype=self.dtype,
            name='gamma'
        )

    def call(self, inputs):
        # Scale the inputs by gamma (broadcasting over batch & spatial dims)
        return inputs * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update({
            "projection_dim": self.projection_dim,
            "init_values": self.init_values,
        })
        return config


class StochasticDepth(layers.Layer):
    def __init__(self, drop_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if (not training) or (self.drop_prob == 0.0):
            return x

        batch_size = tf.shape(x)[0]
        random_tensor = 1.0 - self.drop_prob
        random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=x.dtype)
        binary_tensor = tf.floor(random_tensor)
        output = (x / (1.0 - self.drop_prob)) * binary_tensor
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "drop_prob": self.drop_prob,
        })
        return config


class MyModel(tf.keras.Model):
    """
    A representative ConvNeXt-like model definition with LayerScale and StochasticDepth layers included,
    to allow loading and re-saving ConvNeXt models that use these custom layers.

    NOTE: This is a minimal drop-in reimplementation containing the critical custom layers mentioned in 
    the GitHub issue. The full ConvNeXt architecture is complex and available in keras.applications.convnext.
    Here we focus on providing the LayerScale and StochasticDepth layers enabling loading saved ConvNeXt models.
    """

    def __init__(self, input_shape=(572, 572, 3), num_classes=1000):
        super().__init__()
        self.input_layer = layers.InputLayer(input_shape=input_shape)

        # Example ConvNeXt block skeleton with LayerScale and StochasticDepth
        self.conv = layers.Conv2D(64, kernel_size=7, padding="same")
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.depthwise = layers.DepthwiseConv2D(kernel_size=7, padding="same")
        self.pointwise1 = layers.Dense(128)
        self.act = layers.Activation("gelu")
        self.pointwise2 = layers.Dense(64)
        self.layer_scale = LayerScale(projection_dim=64)
        self.stochastic_depth = StochasticDepth(drop_prob=0.1)
        self.flatten = layers.Flatten()
        self.classifier = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        x = self.input_layer(inputs)
        x = self.conv(x)
        x = self.norm(x)
        x = self.depthwise(x)
        x = self.pointwise1(x)
        x = self.act(x)
        x = self.pointwise2(x)

        x = self.layer_scale(x)
        x = self.stochastic_depth(x, training=training)

        x = self.flatten(x)
        return self.classifier(x)


def my_model_function():
    # Return an instance of MyModel with typical ImageNet scale input shape and 1000 classes.
    return MyModel(input_shape=(572, 572, 3), num_classes=1000)


def GetInput():
    # Return a random float32 tensor of shape (batch, height, width, channels)
    # We use batch=1 for example usage.
    return tf.random.uniform((1, 572, 572, 3), dtype=tf.float32)


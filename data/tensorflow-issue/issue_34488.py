# tf.random.uniform((B, H, W, 3), dtype=tf.float32)  ← Assume input channels=3 as in example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # In the provided example, conv1 was added via append (works fine),
        # conv2 was added via insert (ignored by model due to ListWrapper bug).
        # To fix and unify, wrap all layers properly in tf.keras.layers.LayerList or
        # as attributes. Here, we keep conv1 and conv2 as lists of lists of Conv2D,
        # but we register all layers properly as attributes.
        
        # Use lists to hold sub-lists of layers, but each layer must be registered.
        self.conv1 = [[tf.keras.layers.Conv2D(8, 3, padding='same', name="conv1")]]
        self.conv2 = [[tf.keras.layers.Conv2D(16, 3, padding='same', name="conv2")]]

        # Register the nested layers explicitly to avoid ListWrapper ignoring them.
        # Keras needs layers as attributes or inside tf.keras.layers.LayerList or similar.
        # Here, flatten them and assign as attributes with unique names:
        self._all_layers = []
        for i, conv_list in enumerate(self.conv1):
            for j, layer in enumerate(conv_list):
                setattr(self, f'conv1_{i}_{j}', layer)
                self._all_layers.append(layer)
        for i, conv_list in enumerate(self.conv2):
            for j, layer in enumerate(conv_list):
                setattr(self, f'conv2_{i}_{j}', layer)
                self._all_layers.append(layer)

    def call(self, inputs):
        x = inputs
        # Use conv1[0][0] and conv2[0][0] as per original example
        x = self.conv1[0][0](x)
        x = self.conv2[0][0](x)
        return x

def my_model_function():
    # Return an instance of MyModel (with initialized layers)
    return MyModel()

def GetInput():
    # Return a random input tensor compatible with MyModel
    # The example model expects: (batch, height, width, 3 channels)
    # Without dimensions specified, use some small size like 32x32 batch 1
    return tf.random.uniform((1, 32, 32, 3), dtype=tf.float32)


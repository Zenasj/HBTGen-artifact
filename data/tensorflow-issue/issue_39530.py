# tf.random.uniform((1, 256, 256, 1), dtype=tf.float32) ← Input shape is (256,256,1) grayscale images as per the example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We replicate the start of the ResNet50 architecture but replace the first Conv2D layer with a new one
        # Original ResNet50 first layers (simplified):
        # Input -> ZeroPadding2D -> Conv2D(64, 7x7, strides=2, padding='same') -> BatchNorm -> Activation -> MaxPool
        #
        # We'll create our own starting layers to replace the first convolutional layer to handle single channel inputs.
        #
        # After that, we'll load pretrained ResNet50 without the top,
        # and we'll slice the model starting from the 4th layer onward (skipping original ZeroPadding2D and Conv2D)
        # and use their pretrained weights.
        #
        # This approach mimics taking pretrained ResNet50, removing the first conv (and zero padding) and replacing them
        # with a custom initial conv layer for grayscale input.

        # New first conv layers for single channel input
        self.new_zero_padding = tf.keras.layers.ZeroPadding2D(padding=3)
        self.new_conv = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(7,7), strides=(2,2), padding='same', use_bias=False,
            name='conv1_custom'
        )
        self.new_bn = tf.keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='bn_conv1_custom')
        self.new_activation = tf.keras.layers.Activation('relu')
        self.new_max_pool = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')

        # Load pretrained ResNet50 without the top and weights
        # We need the pretrained weights for layers after the initial conv layer
        base_resnet = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')

        # Extract all layers starting from layer 4 onward:
        # Layers 0-3 in ResNet50 Keras implementation:
        # 0: input_1 (InputLayer)
        # 1: zero_padding2d (ZeroPadding2D)
        # 2: conv1_conv (Conv2D)
        # 3: conv1_bn (BatchNormalization)
        # We want to replace layers 1 and 2 with our custom zero padding and conv for single channel.
        # But we keep the batch norm and all subsequent layers from base ResNet.

        # We'll collect layers from base_resnet starting at 4th layer (index 4), which is conv1_relu in official Keras:
        # Actually typically layer index 4 is conv1_relu activation.
        # So we add batch norm layer from base_resnet as well (layer 3), or replace by our own.
        # To be clear, we'll replicate the forward pass from the point just after our maxpool.

        # To keep it clear and simple, let's create a Sequential for the rest of the ResNet layers from base_resnet starting
        # after the first conv layer and padding.

        # We determine the starting layer index by inspecting the base_resnet layers names:
        # We want to skip original zero padding and conv layers:
        # Let's skip layers with name 'zero_padding2d' and 'conv1_conv'
        # The batchnorm for conv1 is named 'conv1_bn' and activation 'conv1_relu'
        # We want to keep batch norm and activation?

        # To keep the original pretrained batch norm and activation consistent with pretrained weights,
        # let's keep them, so after our conv layer, we use base_resnet.layers[3:] (from 'conv1_bn').

        # For clarity:
        # We'll forward input through our custom new_zero_padding and new_conv layers,
        # then the rest of the base ResNet layers starting from conv1_bn onward.

        # Create a list of layers from layer 3 onward
        self.rest_of_resnet_layers = base_resnet.layers[3:]

        # To execute the rest of the ResNet layers sequentially, use a list of layers and call them one by one in call.
        # We skip the input layer (layer 0).

    def call(self, inputs, training=False):
        x = self.new_zero_padding(inputs)
        x = self.new_conv(x)
        x = self.new_bn(x, training=training)
        x = self.new_activation(x)
        x = self.new_max_pool(x)
        # Now proceed through the rest of pretrained ResNet layers starting from conv1_bn onward

        for layer in self.rest_of_resnet_layers:
            # The layers may have training argument in call if they are BatchNorm or Dropout
            # So we try passing training, else call without it
            try:
                x = layer(x, training=training)
            except TypeError:
                x = layer(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Optionally, build model by calling it once with dummy input
    dummy_input = GetInput()
    _ = model(dummy_input, training=False)
    return model


def GetInput():
    # Return a batch size 1 random tensor shaped (1,256,256,1), dtype float32
    # matches the input shape of the custom grayscale ResNet input
    return tf.random.uniform((1, 256, 256, 1), dtype=tf.float32)


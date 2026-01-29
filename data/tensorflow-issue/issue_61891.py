# tf.random.uniform((1, 7, 7, 4), dtype=tf.float32) ← Inferred input shape from issue description

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D

# Predefined convolution layers with fixed constant kernel initializer as per issue
create_conv_layer1 = Conv2D(filters=4, kernel_size=(3, 3), activation='relu', 
                            trainable=False, kernel_initializer=tf.initializers.Constant(0.5),
                            padding='valid', name='conv1')
create_conv_layer2 = Conv2D(filters=1, kernel_size=(3, 3), activation='relu', 
                            trainable=False, kernel_initializer=tf.initializers.Constant(0.5),
                            padding='valid', name='conv2')


class PatchBasedConv2D(Layer):
    def __init__(self, **kwargs):
        super(PatchBasedConv2D, self).__init__(**kwargs)
        # The patch-based depth-first conv inference layer from the issue
        self.layer_number = 2
        self.output_size = (3, 3, 1)  # final output size after patch-based conv
        self.expand_size = (1, 1)      # used internally
        self.patch_stride = None

    @staticmethod
    def compute_last_patch_size(output_size, kernel_size, stride):
        # Calculate input patch size based on output size, kernel and stride
        input_height = (output_size[0] - 1) * stride[0] + kernel_size[0]
        input_width = (output_size[1] - 1) * stride[1] + kernel_size[1]
        return (input_height, input_width)

    def calculate_patch_count(self, input_size, patch_size, stride):
        # Number of patches in the input spatial dimension
        return ((input_size[0] - patch_size[0]) // stride[0] + 1) * ((input_size[1] - patch_size[1]) // stride[1] + 1)

    def get_current_patch_possition(self, input_size, patch_size, stride, current_round):
        # Compute (row, col) position of current patch based on iteration number
        patches_per_row = (input_size[1] - patch_size[1]) // stride[1] + 1
        pos_y = (current_round // patches_per_row) * stride[0]
        pos_x = (current_round % patches_per_row) * stride[1]
        return (pos_y, pos_x)

    def build(self, input_shape):
        # Assign conv layers, compute patch sizes and strides
        self.conv1 = create_conv_layer1
        self.conv2 = create_conv_layer2

        # Extract kernel sizes and strides of conv layers for computations
        # Both are tuples (height_stride, width_stride)
        self.patch_size_tmp = self.compute_last_patch_size(self.expand_size, self.conv2.kernel_size, self.conv2.strides)
        self.patch_size = self.compute_last_patch_size(self.patch_size_tmp, self.conv1.kernel_size, self.conv1.strides)
        self.patch_stride = self.conv1.strides

    def call(self, inputs):
        inputs_shape = tf.shape(inputs)
        # If conv1 padding was 'same' (issue pad explicitly), add zero padding manually
        if self.conv1.padding == 'same':
            inputs = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')

        input_h = inputs_shape[1]
        input_w = inputs_shape[2]

        num_patch = self.calculate_patch_count((input_h, input_w), self.patch_size, self.patch_stride)
        number_patch_in_row = self.output_size[1] // self.expand_size[1]

        output_feature_map_tmp = None 
        output_feature_map = None

        for current_round in range(num_patch):
            position = self.get_current_patch_possition((input_h, input_w), self.patch_size, self.patch_stride, current_round)
            # Extract patch window
            patch = inputs[:, 
                           position[0]: position[0] + self.patch_size[0], 
                           position[1]: position[1] + self.patch_size[1], :]
            # Depth-first two conv layers inference on patch
            patch_output = self.conv1(patch)
            patch_output = self.conv2(patch_output)

            if output_feature_map_tmp is None:
                output_feature_map_tmp = patch_output
            else:
                # Concatenate patches horizontally
                output_feature_map_tmp = tf.concat([output_feature_map_tmp, patch_output], axis=2)

            # After finishing a row of patches, concatenate vertically
            if (current_round + 1) % number_patch_in_row == 0:
                if output_feature_map is None:
                    output_feature_map = output_feature_map_tmp
                else:
                    output_feature_map = tf.concat([output_feature_map, output_feature_map_tmp], axis=1)
                output_feature_map_tmp = None

        return output_feature_map


class LayerByLayerConv(tf.keras.Model):
    def __init__(self):
        super(LayerByLayerConv, self).__init__()
        # This model applies conv1 to whole input, then conv2 to whole conv1 output
        self.conv1 = create_conv_layer1
        self.conv2 = create_conv_layer2

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Encapsulate both models:
        #  - Patch-based depth-first inference (custom layer)
        #  - Regular layer-by-layer conv (reference)
        self.patch_model = PatchBasedConv2D()
        self.layer_model = LayerByLayerConv()

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # Compute outputs from both models
        output_patch = self.patch_model(inputs)
        output_layer = self.layer_model(inputs)

        # Compare outputs numerically; allow for small tolerance due to floating point
        # Calculate absolute difference
        diff = tf.abs(output_patch - output_layer)
        tolerance = 1e-5
        comparison = tf.reduce_all(diff < tolerance)

        # Return bool if outputs match within tolerance, along with both outputs and diff
        # (to understand differences if needed)
        # Output tuple: (boolean scalar tensor, patch_model output, layer_model output, difference tensor)
        return comparison, output_patch, output_layer, diff


def my_model_function():
    # Return an instance of MyModel, with no special initialization needed
    return MyModel()


def GetInput():
    # Return a random input tensor matching expected shape (1, 7, 7, 4) with float32 dtype
    # Matches the shape as used in the reported issue
    return tf.random.uniform((1, 7, 7, 4), dtype=tf.float32)


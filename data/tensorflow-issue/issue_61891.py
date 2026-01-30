import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

create_conv_layer1 = Conv2D(filters=4, kernel_size=(3, 3), activation='relu', trainable=False, kernel_initializer=tf.initializers.Constant(0.5))
create_conv_layer2 = Conv2D(filters=1, kernel_size=(3, 3), activation='relu', trainable=False, kernel_initializer=tf.initializers.Constant(0.5))

class PatchBasedConv2D(Layer):
    def __init__(self, **kwargs):
        super(PatchBasedConv2D, self).__init__(**kwargs)
        self.layer_number = 2
        self.output_size = (3, 3, 1)
        self.expand_size = (1, 1)
        self.patch_stride = None
    
    @staticmethod
    def compute_last_patch_size(output_size, kernel_size, stride):
        input_height = (output_size[0] - 1) * stride[0] + kernel_size[0]
        input_width = (output_size[1] - 1) * stride[1] + kernel_size[1]
        return (input_height, input_width)
    
    def calculate_patch_count(self, input_size, patch_size, stride):
        return ((input_size[0] - patch_size[0]) // stride[0] + 1) * ((input_size[1] - patch_size[1]) // stride[1] + 1)
    
    def get_current_patch_possition(self, input_size, patch_size, stride, current_round):
        return ((current_round // ((input_size[1] - patch_size[1]) // stride[1] + 1)) * stride[0],
              (current_round % ((input_size[1] - patch_size[1]) // stride[1] + 1)) * stride[1])

    def build(self, input_shape):
        with tf.device('/CPU:0'):
            self.conv1 = create_conv_layer1
            self.conv2 = create_conv_layer2
            self.patch_size_tmp = self.compute_last_patch_size(self.expand_size, self.conv2.kernel_size, self.conv2.strides)
            self.patch_size = self.compute_last_patch_size(self.patch_size_tmp, self.conv1.kernel_size, self.conv1.strides)
            self.patch_stride = self.conv1.strides

    def call(self, inputs):
        if self.conv1.padding == 'same':
            inputs = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
            
        num_patch = self.calculate_patch_count((inputs.shape[1], inputs.shape[2]), self.patch_size, self.patch_stride)
        
        number_patch_in_row = int(self.output_size[1] // self.expand_size[1])
        output_feature_map_tmp = None 
        output_feature_map = None
        
        for current_round in range(num_patch):
            position = get_current_patch_possition((inputs.shape[1], inputs.shape[2]), self.patch_size, self.patch_stride, current_round)
            patch_output = self.conv1(inputs[:, position[0]: position[0] + self.patch_size[0], position[1]: position[1] + self.patch_size[1], :])
            patch_output = self.conv2(patch_output)
            
            if output_feature_map_tmp is None:
                output_feature_map_tmp = patch_output
            else:
                output_feature_map_tmp = tf.concat([output_feature_map_tmp, patch_output], axis=2)
            
            if (current_round + 1) % number_patch_in_row == 0 and current_round != 0:
                if output_feature_map is None:
                    output_feature_map = output_feature_map_tmp
                else:
                    output_feature_map = tf.concat([output_feature_map, output_feature_map_tmp], axis=1)
                output_feature_map_tmp = None
        
        return output_feature_map

custom_layer = PatchBasedConv2D()
input_tensor = tf.keras.layers.Input(shape = input_tensor.shape)
output_tensor = custom_layer(input)
model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
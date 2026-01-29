# tf.random.uniform((B, 128, 128, 50, 1), dtype=tf.float32)

import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Dropout, Input, Concatenate, Add
from tensorflow.keras import Model

class MyModel(tf.keras.Model):
    def __init__(self, input_shape=(128,128,50,1), n_class=3, multilabel=False):
        super().__init__()
        self.input_shape_ = input_shape
        self.n_class = n_class
        self.multilabel = multilabel
        
        # Choose final activation function based on multilabel flag
        self.activation_fn = 'sigmoid' if multilabel else 'softmax'

        # Initial convolution
        self.conv_1 = Conv3D(filters=64, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation='relu')

        # Spatial reduction blocks (2)
        self.spatial_reduction_block_1 = self._build_spatial_reduction_block('spatial_reduction_block_1')
        self.spatial_reduction_block_2 = self._build_spatial_reduction_block('spatial_reduction_block_2')
        
        # Residual convolution blocks (2)
        self.residual_convolution_block_1 = self._build_residual_convolution_block('residual_convolution_block_1')
        self.residual_convolution_block_2 = self._build_residual_convolution_block('residual_convolution_block_2')

        # Later convolutions
        self.conv_2 = Conv3D(filters=512, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation='relu')
        self.maxpool_1 = MaxPool3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid')
        self.conv_3 = Conv3D(filters=1024, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation='relu')
        self.maxpool_2 = MaxPool3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid')
        self.flatten = Flatten()
        self.dropout_1 = Dropout(rate=0.2)
        self.dense_1 = Dense(512, activation='sigmoid')
        self.dropout_2 = Dropout(rate=0.2)
        self.outputs_layer = Dense(n_class, activation=self.activation_fn, name='outputs')

    def call(self, inputs, training=False):
        x = self.conv_1(inputs)
        x = self.spatial_reduction_block_1(x)
        x = self.residual_convolution_block_1(x)
        x = self.spatial_reduction_block_2(x)
        x = self.residual_convolution_block_2(x)
        x = self.conv_2(x)
        x = self.maxpool_1(x)
        x = self.conv_3(x)
        x = self.maxpool_2(x)
        x = self.flatten(x)
        x = self.dropout_1(x, training=training)
        x = self.dense_1(x)
        x = self.dropout_2(x, training=training)
        out = self.outputs_layer(x)
        return out

    def _build_spatial_reduction_block(self, block_name):
        # Builds a spatial reduction block as a keras Model for reuse
        # Uses functional style within call method, implemented here as a subclassed layer
        # This block expects input tensor of shape (..., D, H, W, C)
        # Filters inferred from input channels. 
        # Implemented here as sublayers because filters are dynamic: they depend on actual tensor shape
        
        class SpatialReductionBlock(tf.keras.layers.Layer):
            def __init__(self, name):
                super().__init__(name=name)
                # Layers use filters that can vary, so actual creation is delayed to call
                # We'll create layers dynamically first call (lazy build)
                self._built = False

            def build(self, input_shape):
                filters = input_shape[-1]
                # Create layers
                self.maxpool = MaxPool3D(pool_size=(2,2,2), strides=(2,2,2), padding='same')
                self.conv_a_0 = Conv3D(filters=filters//4, kernel_size=(3,3,3), strides=(2,2,2), padding='same', activation='relu')
                self.conv_b_0 = Conv3D(filters=filters, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation='relu')
                self.conv_c_0 = Conv3D(filters=filters, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation='relu')
                self.conv_b_1 = Conv3D(filters=(5*filters)//16, kernel_size=(3,3,3), strides=(2,2,2), padding='same', activation='relu')
                self.conv_c_1 = Conv3D(filters=(5*filters)//16, kernel_size=(3,3,3), strides=(1,1,1), padding='same', activation='relu')
                self.conv_c_2 = Conv3D(filters=(7*filters)//16, kernel_size=(3,3,3), strides=(2,2,2), padding='same', activation='relu')
                self.concat = Concatenate()
                self._built = True

            def call(self, inputs):
                if not self._built:
                    self.build(inputs.shape)
                maxpool = self.maxpool(inputs)
                conv_a_0 = self.conv_a_0(inputs)
                conv_b_0 = self.conv_b_0(inputs)
                conv_c_0 = self.conv_c_0(inputs)
                conv_b_1 = self.conv_b_1(conv_b_0)
                conv_c_1 = self.conv_c_1(conv_c_0)
                conv_c_2 = self.conv_c_2(conv_c_1)
                return self.concat([maxpool, conv_a_0, conv_b_1, conv_c_2])

        return SpatialReductionBlock(block_name)

    def _build_residual_convolution_block(self, block_name):
        # Similar approach to spatial_reduction_block, using dynamic filters from input shape
        
        class ResidualConvolutionBlock(tf.keras.layers.Layer):
            def __init__(self, name):
                super().__init__(name=name)
                self._built = False

            def build(self, input_shape):
                filters = input_shape[-1]
                self.conv_a_0 = Conv3D(filters=filters//2, kernel_size=(3,3,3), strides=(1,1,1), padding='same', activation='relu')
                self.conv_b_0 = Conv3D(filters=filters//2, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation='relu')
                self.conv_c_0 = Conv3D(filters=filters//2, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation='relu')
                self.conv_b_1 = Conv3D(filters=filters//2, kernel_size=(3,3,3), strides=(1,1,1), padding='same', activation='relu')
                self.conv_c_1 = Conv3D(filters=filters//2, kernel_size=(3,3,3), strides=(1,1,1), padding='same', activation='relu')
                self.conv_c_2 = Conv3D(filters=filters//2, kernel_size=(3,3,3), strides=(1,1,1), padding='same', activation='relu')
                self.concat = Concatenate()
                self.conv_d_0 = Conv3D(filters=filters, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation='relu')
                self.add = Add()
                self._built = True

            def call(self, inputs):
                if not self._built:
                    self.build(inputs.shape)
                conv_a_0 = self.conv_a_0(inputs)
                conv_b_0 = self.conv_b_0(inputs)
                conv_c_0 = self.conv_c_0(inputs)
                conv_b_1 = self.conv_b_1(conv_b_0)
                conv_c_1 = self.conv_c_1(conv_c_0)
                conv_c_2 = self.conv_c_2(conv_c_1)
                concat_output = self.concat([conv_a_0, conv_b_1, conv_c_2])
                conv_d_0 = self.conv_d_0(concat_output)
                return self.add([conv_d_0, inputs])

        return ResidualConvolutionBlock(block_name)


def my_model_function():
    # Return an instance of MyModel initialized with default input shape and classes (as per the original)
    return MyModel(input_shape=(128,128,50,1), n_class=3, multilabel=False)


def GetInput():
    # Return a random input tensor with shape (batch_size, 128, 128, 50, 1)
    # Use batch size of 2 as an example (can be any positive integer)
    batch_size = 2
    input_shape = (batch_size, 128, 128, 50, 1)
    # Use float32 dtype, consistent with typical image data tensors
    random_input = tf.random.uniform(input_shape, dtype=tf.float32)
    return random_input


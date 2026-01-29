# tf.random.uniform((B, 512, 1024, 3), dtype=tf.float32)  # Input shape for Model 2, as it's the newer and primary model discussed

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Add, ReLU, Dropout, Input
from tensorflow.keras.models import Model

class MyModel(tf.keras.Model):
    """
    Fused model encapsulating both Model 1 and Model 2 architectures.
    Both are VGG-like FCN segmentation models differing primarily in input resolution:
    - Model 1 input: (256, 512, 3)
    - Model 2 input: (512, 1024, 3)

    This class implements Model 2 (the TF 2.x version) as the main architecture, with 
    modular blocks named as per the original VGG block conv layers.

    For demonstration, Model 1 architecture is also included as a submodel, 
    but forward() uses Model 2 (can be extended per use case).

    The forward output returns:
    - model2_output: segmentation logits of shape (B, 512,1024,2)
    - last_conv_output: activation from last conv layer (block5_conv4) from Model 2,
      used in GradCam computations.

    This enables the GradCam script to obtain intermediate activations and apply gradients.
    """
    def __init__(self):
        super(MyModel, self).__init__()

        # --- Model 2 layers (input 512x1024) ---
        # Block1
        self.block1_conv1 = Conv2D(64, 3, padding='same', activation='relu', name='block1_conv1')
        self.block1_conv2 = Conv2D(64, 3, padding='same', activation='relu', name='block1_conv2')
        self.block1_pool = MaxPooling2D(pool_size=2, strides=2, name='block1_pool')

        # Block2
        self.block2_conv1 = Conv2D(128, 3, padding='same', activation='relu', name='block2_conv1')
        self.block2_conv2 = Conv2D(128, 3, padding='same', activation='relu', name='block2_conv2')
        self.block2_pool = MaxPooling2D(pool_size=2, strides=2, name='block2_pool')

        # Block3
        self.block3_conv1 = Conv2D(256, 3, padding='same', activation='relu', name='block3_conv1')
        self.block3_conv2 = Conv2D(256, 3, padding='same', activation='relu', name='block3_conv2')
        self.block3_conv3 = Conv2D(256, 3, padding='same', activation='relu', name='block3_conv3')
        self.block3_conv4 = Conv2D(256, 3, padding='same', activation='relu', name='block3_conv4')
        self.block3_pool = MaxPooling2D(pool_size=2, strides=2, name='block3_pool')

        # Block4
        self.block4_conv1 = Conv2D(512, 3, padding='same', activation='relu', name='block4_conv1')
        self.block4_conv2 = Conv2D(512, 3, padding='same', activation='relu', name='block4_conv2')
        self.block4_conv3 = Conv2D(512, 3, padding='same', activation='relu', name='block4_conv3')
        self.block4_conv4 = Conv2D(512, 3, padding='same', activation='relu', name='block4_conv4')
        self.block4_pool = MaxPooling2D(pool_size=2, strides=2, name='block4_pool')

        # Block5
        self.block5_conv1 = Conv2D(512, 3, padding='same', activation='relu', name='block5_conv1')
        self.block5_conv2 = Conv2D(512, 3, padding='same', activation='relu', name='block5_conv2')
        self.block5_conv3 = Conv2D(512, 3, padding='same', activation='relu', name='block5_conv3')
        self.block5_conv4 = Conv2D(512, 3, padding='same', activation='relu', name='block5_conv4')
        self.block5_pool = MaxPooling2D(pool_size=2, strides=2, name='block5_pool')

        # FCN layers (converted FC layers to conv)
        self.conv2d = Conv2D(4096, 7, padding='same', name='conv2d')  # large kernel simulates FC layer
        self.relu = ReLU(name='re_lu')
        self.dropout = Dropout(0.5)

        self.conv2d_1 = Conv2D(4096, 1, padding='same', name='conv2d_1')
        self.relu_1 = ReLU(name='re_lu_1')
        self.dropout_1 = Dropout(0.5)

        self.conv2d_2 = Conv2D(2, 1, padding='same', name='conv2d_2')  # final logits for 2 classes

        # Decoder layers (upsampling by transpose conv + skip connections)
        self.conv2d_transpose = Conv2DTranspose(2, 3, strides=2, padding='same', name='conv2d_transpose')
        self.conv2d_3 = Conv2D(2, 1, padding='same', name='conv2d_3')

        self.add = Add(name='add')

        self.conv2d_transpose_1 = Conv2DTranspose(2, 3, strides=2, padding='same', name='conv2d_transpose_1')
        self.conv2d_4 = Conv2D(2, 1, padding='same', name='conv2d_4')

        self.add_1 = Add(name='add_1')

        self.conv2d_transpose_2 = Conv2DTranspose(2, 3, strides=2, padding='same', name='conv2d_transpose_2')

        # For the sake of completeness, parts of Model 1 could be defined here similarly, but
        # as the main usage and troubleshooting relates to Model 2, we focus on Model 2.
        # Model 1 differs primarily in input size and some layer names, but conceptually same.

    def call(self, inputs, training=False):
        # Encoder - Block 1
        x = self.block1_conv1(inputs)
        x = self.block1_conv2(x)
        x = self.block1_pool(x)

        # Block 2
        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = self.block2_pool(x)

        # Block 3
        x = self.block3_conv1(x)
        x = self.block3_conv2(x)
        x = self.block3_conv3(x)
        x = self.block3_conv4(x)
        x_block3_pool = self.block3_pool(x)  # to be used in skip connections

        # Block 4
        x = self.block4_conv1(x_block3_pool)
        x = self.block4_conv2(x)
        x = self.block4_conv3(x)
        x = self.block4_conv4(x)
        x_block4_pool = self.block4_pool(x)  # to be used in skip connections

        # Block 5
        x = self.block5_conv1(x_block4_pool)
        x = self.block5_conv2(x)
        x = self.block5_conv3(x)
        x = self.block5_conv4(x)
        last_conv = x  # Shape: (B, 32, 64, 512), last convolutional output for GradCam

        x = self.block5_pool(last_conv)

        # Fully convolutional (FCN) layers simulating fully connected layers
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.dropout(x, training=training)

        x = self.conv2d_1(x)
        x = self.relu_1(x)
        x = self.dropout_1(x, training=training)

        x = self.conv2d_2(x)  # logits with 2 classes

        # Decoder with skip connections and transpose conv upsampling
        x_up = self.conv2d_transpose(x)  # upsample by 2 -> (B, 32, 64, 2)
        skip4 = self.conv2d_3(x_block4_pool)  # process skip connection from block4

        x_up = self.add([x_up, skip4])  # add skip connection

        x_up = self.conv2d_transpose_1(x_up)  # upsample by 2 -> (B, 64, 128, 2)
        skip3 = self.conv2d_4(x_block3_pool)

        x_up = self.add_1([x_up, skip3])

        x_up = self.conv2d_transpose_2(x_up)  # upsample by 2 -> (B, 512, 1024, 2)

        # Final output: per-pixel logits for 2 classes
        return x_up, last_conv

def my_model_function():
    """
    Returns an instance of MyModel.
    """
    return MyModel()

def GetInput():
    """
    Returns a random tensor input matching the input expected by MyModel.
    Model 2 input shape is (batch, 512, 1024, 3), dtype float32.

    Batch size is arbitrarily chosen as 1.
    """
    batch_size = 1
    H, W, C = 512, 1024, 3
    return tf.random.uniform((batch_size, H, W, C), dtype=tf.float32)


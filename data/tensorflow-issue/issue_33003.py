# tf.random.uniform((64, 200, 200, 3), dtype=tf.float32) ← inferred input shape and type from batch_size and img_input shape

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Architecture parameters
        self.image_size = 200
        self.nClasses = 6
        self.n = 32 * 5  # =160
        self.nfmp_block1 = 64
        self.nfmp_block2 = 128
        self.IMAGE_ORDERING = "channels_last"
        
        # Encoder Block 1
        self.block1_conv1 = tf.keras.layers.Conv2D(
            self.nfmp_block1, (3, 3), activation='relu', padding='same', data_format=self.IMAGE_ORDERING, name='block1_conv1')
        self.block1_conv2 = tf.keras.layers.Conv2D(
            self.nfmp_block1, (3, 3), activation='relu', padding='same', data_format=self.IMAGE_ORDERING, name='block1_conv2')
        self.block1_pool = tf.keras.layers.MaxPooling2D(
            (2, 2), strides=(2, 2), data_format=self.IMAGE_ORDERING, name='block1_pool')

        # Encoder Block 2
        self.block2_conv1 = tf.keras.layers.Conv2D(
            self.nfmp_block2, (3, 3), activation='relu', padding='same', data_format=self.IMAGE_ORDERING, name='block2_conv1')
        self.block2_conv2 = tf.keras.layers.Conv2D(
            self.nfmp_block2, (3, 3), activation='relu', padding='same', data_format=self.IMAGE_ORDERING, name='block2_conv2')
        self.block2_pool = tf.keras.layers.MaxPooling2D(
            (2, 2), strides=(2, 2), data_format=self.IMAGE_ORDERING, name='block2_pool')

        # Bottleneck layers
        # Kernel size is (image_size/4, image_size/4) = (50, 50) due to two 2x2 pools halving dims twice from 200->100->50
        self.bottleneck_1 = tf.keras.layers.Conv2D(
            self.n, (self.image_size // 4, self.image_size // 4), activation='relu', padding='same', data_format=self.IMAGE_ORDERING, name='bottleneck_1')
        self.bottleneck_2 = tf.keras.layers.Conv2D(
            self.n, (1, 1), activation='relu', padding='same', data_format=self.IMAGE_ORDERING, name='bottleneck_2')

        # Decoder / Upsampling via Conv2DTranspose
        # Upsamples by stride 4 to restore original spatial dims 50x50 -> 200x200
        self.upsample_2 = tf.keras.layers.Conv2DTranspose(
            self.nClasses, kernel_size=(4, 4), strides=(4, 4), use_bias=False, name='upsample_2', data_format=self.IMAGE_ORDERING)

        # Reshape layer to produce shape (image_size * image_size * nClasses, 1) for temporal weighting
        self.reshape = tf.keras.layers.Reshape((self.image_size * self.image_size * self.nClasses, 1))

    def call(self, inputs, training=None):
        x = self.block1_conv1(inputs)
        x = self.block1_conv2(x)
        block1 = self.block1_pool(x)
        
        x = self.block2_conv1(block1)
        x = self.block2_conv2(x)
        x = self.block2_pool(x)
        
        o = self.bottleneck_1(x)
        o = self.bottleneck_2(o)
        
        output = self.upsample_2(o)
        output = self.reshape(output)
        return output

def my_model_function():
    # Instantiate the model and compile similarly to the original snippet:
    model = MyModel()
    
    # Note: The original code uses an external RAdam optimizer from keras_radam with parameters
    # Since keras_radam is not standard, use the built-in Adam optimizer as a placeholder.
    # Users should replace this with their preferred RAdam implementation if needed.
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    model.compile(
        optimizer=optimizer, 
        loss='mse', 
        # sample_weight_mode='temporal' is deprecated in TF2; assuming no special weight mode
    )
    return model

def GetInput():
    # Generate random input tensor with shape (batch_size, 200, 200, 3)
    # batch_size=64 from original code snippet
    batch_size = 64
    image_size = 200
    # Random uniform floats between 0 and 1
    return tf.random.uniform((batch_size, image_size, image_size, 3), dtype=tf.float32)


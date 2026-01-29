# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32) <- Input assumed batch size B, image size 224x224 with 3 channels (RGB)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Load VGG16 base model without top classification layers and pretrained on ImageNet
        self.vgg16_base = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
        # Extract output from last convolutional layer of VGG16
        self.vgg16_feature_extractor = tf.keras.Model(
            inputs=self.vgg16_base.input,
            outputs=self.vgg16_base.layers[-1].output
        )
        
        # Similarly, load InceptionV3 base model without top classification layers pretrained on ImageNet
        self.inception_base = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        self.inception_feature_extractor = tf.keras.Model(
            inputs=self.inception_base.input,
            outputs=self.inception_base.layers[-1].output
        )
    
    @tf.function(jit_compile=True)
    def call(self, inputs):
        """
        Forward pass:
        inputs: batch of images of shape (B, 224, 224, 3), dtype float32, preprocessed for VGG16
        Returns:
          A boolean tensor indicating whether the features extracted by VGG16 and InceptionV3 models
          are close within a tolerance (L2 norm difference < threshold).
          This encapsulates the comparison logic discussed in the issue.
        """
        # Compute features from VGG16
        vgg_features = self.vgg16_feature_extractor(inputs)
        # Reshape VGG16 features to (batch_size, num_features, channels)
        vgg_features_reshaped = tf.reshape(vgg_features, 
                                           shape=(tf.shape(vgg_features)[0], -1, tf.shape(vgg_features)[-1]))
        
        # Resize inputs to InceptionV3 expected size (299x299)
        inputs_inception = tf.image.resize(inputs, (299, 299))
        # Preprocess inputs for InceptionV3
        inputs_inception = tf.keras.applications.inception_v3.preprocess_input(inputs_inception)
        # Extract features using InceptionV3
        inception_features = self.inception_feature_extractor(inputs_inception)
        # Reshape Inception features similarly
        inception_features_reshaped = tf.reshape(inception_features, 
                                                 shape=(tf.shape(inception_features)[0], -1, tf.shape(inception_features)[-1]))
        
        # To compare shapes, we will reduce both feature tensors along features dimension by mean pooling
        vgg_pooled = tf.reduce_mean(vgg_features_reshaped, axis=1)
        inception_pooled = tf.reduce_mean(inception_features_reshaped, axis=1)
        
        # Compute L2 distance between pooled features
        diff = tf.norm(vgg_pooled - inception_pooled, axis=1)
        
        # Define tolerance threshold for closeness
        threshold = 1e-2
        
        # Boolean tensor: True if difference less than threshold, False otherwise
        close_enough = diff < threshold
        
        return close_enough

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    """
    Returns a batch of random image tensor matching VGG16 input specs:
    - shape: (batch_size, 224, 224, 3)
    - dtype: float32
    - values scaled as expected by VGG16 preprocess_input, which expects pixels in ‘BGR’ format with zero-center by mean subtraction.
    
    Since we're generating random data, apply VGG16 preprocess_input to simulate similar preprocessing steps:
    The typical preprocessing subtracts mean RGB values, but given random uniform inputs in [0, 255), 
    we prepare accordingly.
    """
    batch_size = 4  # example batch size
    
    # Generate random uint8 images in [0, 255], shape (B, H, W, C)
    images_uint8 = tf.random.uniform(shape=(batch_size, 224, 224, 3), minval=0, maxval=256, dtype=tf.int32)
    images_uint8 = tf.cast(images_uint8, tf.uint8)
    
    # Convert to float32
    images_float = tf.cast(images_uint8, tf.float32)
    
    # Use VGG16 preprocess_input - this will convert RGB to BGR and zero center by Imagenet means
    images_preprocessed = tf.keras.applications.vgg16.preprocess_input(images_float)
    
    return images_preprocessed


# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32) ← Input images are 224x224 RGB images

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model acts as a wrapper to instantiate several ImageNet pre-trained classification models:
    ResNet50, DenseNet121, and MobileNetV2 from tf.keras.applications. It applies the appropriate
    preprocessing (normalization to [-1, 1]) internally to input images before feeding each model.

    The forward pass returns a dictionary of logits from each model.
    """

    def __init__(self):
        super().__init__()

        # Instantiate keras.applications models with imagenet weights and include_top=True to get logits
        # pooling='max' is used in the original issue for evaluation, but here we use default global pooling 'avg'
        # for ease. The input shape is assumed fixed to (224, 224, 3).
        self.resnet50 = tf.keras.applications.ResNet50(
            include_top=True, weights='imagenet', input_shape=(224, 224, 3), pooling=None)
        self.densenet121 = tf.keras.applications.DenseNet121(
            include_top=True, weights='imagenet', input_shape=(224, 224, 3), pooling=None)
        self.mobilenetv2 = tf.keras.applications.MobileNetV2(
            include_top=True, weights='imagenet', input_shape=(224, 224, 3), pooling=None)

        # Preprocessing function: Keras preprocess_input normalizes input images in range [0,255] to [-1,1].
        # We will assume input images are in [0,1] float32 range and rescale accordingly before feeding models.
        # This matches the recommended approach from keras.applications.
        self.preprocess = tf.keras.applications.imagenet_utils.preprocess_input

    def call(self, inputs, training=False):
        """
        Args:
          inputs: input tensor with shape (B, 224, 224, 3), float32, expected range [0,1].

        Returns:
          dict mapping model name to logits tensor of shape (B, 1000)
        """
        # Because keras.applications expect images pixel values in range [0, 255], preprocess_input handles
        # this including mean subtraction and scaling to [-1, 1].
        # But note inputs come normalized to [0,1], so multiply by 255 first.
        x = inputs * 255.0

        # preprocess_input applies mode='tf' logic: scales pixels to [-1,1]
        x_preprocessed = self.preprocess(x, mode='tf')

        # Forward pass through each model
        resnet_logits = self.resnet50(x_preprocessed, training=training)
        densenet_logits = self.densenet121(x_preprocessed, training=training)
        mobilenet_logits = self.mobilenetv2(x_preprocessed, training=training)

        outputs = {
            'ResNet50': resnet_logits,
            'DenseNet121': densenet_logits,
            'MobileNetV2': mobilenet_logits,
        }
        return outputs

def my_model_function():
    # Return an instance of MyModel with pretrained weights loaded.
    # The model is ready for inference with inputs normalized in [0,1].
    return MyModel()

def GetInput():
    """
    Return a random input tensor compatible with MyModel.

    Since the models expect input shape (B, 224, 224, 3),
    and input pixel range [0,1], generate random float32 tensor.
    """
    batch_size = 4  # Arbitrary batch size for demonstration
    height = 224
    width = 224
    channels = 3
    # Uniform random floats between 0 and 1
    return tf.random.uniform((batch_size, height, width, channels), minval=0.0, maxval=1.0, dtype=tf.float32)


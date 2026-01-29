# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Input shape inferred as typical image tensor batch

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, base_model_name="mobilenetv2", num_classes=1):
        super().__init__()
        # Based on the issue, the model uses a base pretrained CNN for feature extraction,
        # followed by a final Dense layer with sigmoid activation for binary classification.
        # The issue context referenced MobileNetV2, InceptionV3, and VGG16.
        # Here, we'll fuse these base models as submodules for comparison.
        # The forward will return the differences between outputs to reflect benchmark scenario.

        # Load three base models with ImageNet weights, without top classification layers
        self.base_models = {}

        # MobileNetV2
        mobilenetv2 = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights='imagenet',
            pooling='avg',  # Global average pooling output vector
            input_shape=(224, 224, 3),
        )
        self.base_models['mobilenetv2'] = mobilenetv2

        # InceptionV3
        inceptionv3 = tf.keras.applications.InceptionV3(
            include_top=False,
            weights='imagenet',
            pooling='avg',
            input_shape=(299, 299, 3),
        )
        self.base_models['inceptionv3'] = inceptionv3

        # VGG16
        vgg16 = tf.keras.applications.VGG16(
            include_top=False,
            weights='imagenet',
            pooling='avg',
            input_shape=(224, 224, 3),
        )
        self.base_models['vgg16'] = vgg16

        # A dictionary of Dense layers for each base model to produce final output
        self.dense_layers = {
            'mobilenetv2': tf.keras.layers.Dense(num_classes, activation='sigmoid'),
            'inceptionv3': tf.keras.layers.Dense(num_classes, activation='sigmoid'),
            'vgg16': tf.keras.layers.Dense(num_classes, activation='sigmoid'),
        }

        # Note:
        # Inputs to InceptionV3 are 299x299, MobileNetV2 and VGG16 are 224x224 as typical.
        # To unify inputs, we'll assume input shape is (299,299,3),
        # resize inputs for MobileNetV2 and VGG16 inside the call method.

        # Image resizing layers for base models (except Inception)
        self.resize_mobilenetv2 = tf.keras.layers.Resizing(224, 224)
        self.resize_vgg16 = tf.keras.layers.Resizing(224, 224)

    def call(self, inputs, training=False):
        # inputs shape: (B, 299, 299, 3) - assume this to support largest base input

        # Preprocess inputs for each base model according to their expected preprocess
        # Using tf.keras.applications preprocessing functions:
        mobilenet_input = self.resize_mobilenetv2(inputs)
        mobilenet_input = tf.keras.applications.mobilenet_v2.preprocess_input(mobilenet_input)

        inception_input = tf.keras.applications.inception_v3.preprocess_input(inputs)  # 299x299 no resize needed

        vgg_input = self.resize_vgg16(inputs)
        vgg_input = tf.keras.applications.vgg16.preprocess_input(vgg_input)

        # Extract features
        mobilenet_features = self.base_models['mobilenetv2'](mobilenet_input, training=training)
        inception_features = self.base_models['inceptionv3'](inception_input, training=training)
        vgg_features = self.base_models['vgg16'](vgg_input, training=training)

        # Final output predictions through respective dense layers
        mobilenet_pred = self.dense_layers['mobilenetv2'](mobilenet_features)
        inception_pred = self.dense_layers['inceptionv3'](inception_features)
        vgg_pred = self.dense_layers['vgg16'](vgg_features)

        # Fuse the three outputs into a single tensor
        # For demonstration, output a concatenation (3,1) tensor of predictions per sample
        preds = tf.concat([mobilenet_pred, inception_pred, vgg_pred], axis=1)

        # In a benchmarking scenario, one might want differences or comparisons:
        # For example, return the absolute difference between MobileNetV2 and others
        diff_inception = tf.abs(mobilenet_pred - inception_pred)
        diff_vgg = tf.abs(mobilenet_pred - vgg_pred)

        # Final output returns a dict-like structure (using a tf.Tensor dictionary not supported,
        # so we return concatenated tensor: [mobilenet_pred, inception_pred, vgg_pred, diff_inception, diff_vgg])
        # Shape: (batch_size, 5)
        output = tf.concat([mobilenet_pred, inception_pred, vgg_pred, diff_inception, diff_vgg], axis=1)

        return output

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Return a batch of random input images shaped for InceptionV3 input (299,299,3)
    # Use dtype float32 since model preprocessing expects float
    batch_size = 4  # arbitrary batch size
    input_shape = (batch_size, 299, 299, 3)
    # Random values in [0, 255), since keras preprocess_input expects images in RGB 0-255 range
    input_tensor = tf.random.uniform(input_shape, minval=0, maxval=255, dtype=tf.float32)
    return input_tensor


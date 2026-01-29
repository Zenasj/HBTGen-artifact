# tf.random.uniform((1, 224, 224, 3), dtype=tf.float32) ← input shape for the models used (VGG16, ResNet50, MobileNetV2, or custom Conv2D+BN)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, model_name="VGG16", random_bn_params=False):
        super().__init__()
        self.model_name = model_name
        self.random_bn_params = random_bn_params

        # Build either a pretrained Keras model or a custom Conv2D + BatchNorm model
        if self.model_name == "ResNet50":
            self.base_model = tf.keras.applications.ResNet50(
                input_shape=(224, 224, 3),
                include_top=False,
                pooling="avg")
        elif self.model_name == "MobileNetV2_1.0":
            self.base_model = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                pooling="avg")
        elif self.model_name == "VGG16":
            self.base_model = tf.keras.applications.VGG16(
                input_shape=(224, 224, 3),
                include_top=False,
                pooling="avg")
        else:
            # Custom Conv2D + BatchNorm model with optional random BN parameter init
            if self.random_bn_params:
                params = {
                    "beta_initializer": tf.random_normal_initializer(),
                    "gamma_initializer": tf.random_normal_initializer(),
                    "moving_mean_initializer": tf.random_normal_initializer(),
                    "moving_variance_initializer": tf.random_normal_initializer()
                }
            else:
                params = dict()
            self.conv = tf.keras.layers.Conv2D(1, (3, 3), use_bias=False)
            self.bn = tf.compat.v1.keras.layers.BatchNormalization(**params)
            # Input layer is not explicitly stored; handle in call
            
    def call(self, inputs, training=False):
        if self.model_name in {"ResNet50", "MobileNetV2_1.0", "VGG16"}:
            # Forward pass through the base pretrained model
            return self.base_model(inputs, training=training)
        else:
            # Custom Conv2D + BN
            x = self.conv(inputs)
            x = self.bn(x, training=training)
            return x


def my_model_function():
    # Using default values from the provided example
    # Note: In the original issue, MODEL_NAME="VGG161" was a typo; change to "VGG16" or others
    model_name = "VGG16"  # Options: "ResNet50", "MobileNetV2_1.0", "VGG16", or other for Conv2D+BN
    random_bn_params = True  # Whether to initialize BN layers with random parameters (True matches original example)
    model = MyModel(model_name=model_name, random_bn_params=random_bn_params)
    # We can compile to set optimizer and loss as in original code (optimizer with lr=0 and decay=1 to stop training)
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0, decay=1.0),
        loss='mean_squared_error'
    )
    return model


def GetInput():
    # Return a fixed input tensor corresponding to the input shape of the models used
    # Original code used tf.ones((1,224,224,3)) as fixed model input
    return tf.ones((1, 224, 224, 3), dtype=tf.float32)


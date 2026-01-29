# tf.random.uniform((1, 224, 224, 3), dtype=tf.float32) ← Input shape inferred from MobileNetV3Small example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Use MobileNetV3Small backbone similar to the issue example
        # include_top=False and pooling='max' to get a feature vector output
        self.mobilenet = tf.keras.applications.MobileNetV3Small(
            input_shape=[224, 224, 3],
            alpha=1.0,
            minimalistic=False,
            include_top=False,
            weights="imagenet",
            pooling="max",
            dropout_rate=0.2,
            classifier_activation=None,
            include_preprocessing=True,
        )

    def call(self, inputs, training=False):
        # Forward pass just through MobileNet backbone
        x = self.mobilenet(inputs, training=training)
        return x

def my_model_function():
    # Return an instance of MyModel, pretrained MobileNet backbone is loaded by default
    return MyModel()

def GetInput():
    # Return a single batch of input tensor with shape matching MobileNet V3 input format
    # Use uniform random input in [0,1) as plausible input data
    # dtype float32 matches expected input type
    return tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)


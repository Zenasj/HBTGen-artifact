# tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)  ← Assumed typical input shape for DenseNet121 (default input size 224x224 RGB images)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embed DenseNet121 similar to the issue example,
        # configured with no pretrained weights, 3 output classes, softmax activation
        self.densenet = tf.keras.applications.DenseNet121(
            include_top=True,
            weights=None,
            classes=3,
            classifier_activation='softmax'
        )

    def call(self, inputs, training=False):
        # Just forward through DenseNet121
        return self.densenet(inputs, training=training)

def my_model_function():
    # Return an instance of the DenseNet121-based model
    return MyModel()

def GetInput():
    # Return a random tensor with the DenseNet121 expected input shape
    # DenseNet121 expects images sized 224x224 with 3 channels and batch size of 1
    return tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)


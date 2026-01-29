# tf.random.uniform((B, H, W, C), dtype=tf.float32) <- Typical shape for ImageNet Inception input: batch 1, 299x299 RGB images

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    Reconstructed model representing the Inception v3 image classification model
    loaded from the TensorFlow pretrained graph used in the original issue.
    Since the original example loads the model from a GraphDef proto and runs inference,
    here we emulate the key parts as a Keras model using TensorFlow 2.x, for demonstration.
    
    NOTE: The real original code used frozen GraphDef files and sessions (TF1 style).
    This model here is a conceptual replacement suitable for TF2 and XLA compilation.
    
    No weights are loaded here — the original issue focused on GPU memory behavior,
    so this placeholder will accept inputs of expected image shape (batch, 299, 299, 3)
    and output a 1000-class prediction with random weights.
    
    This meets the requirement for a runnable TF2 model with inference input/output signature.
    """

    def __init__(self):
        super().__init__()
        # Emulate some layers of Inception v3 roughly:
        # Start with some Conv layers, pooling, then a dense 1000-class output.
        # These are placeholders; original weights and architecture are complex.
        self.conv1 = tf.keras.layers.Conv2D(32, 3, strides=2, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu')
        self.maxpool = tf.keras.layers.MaxPooling2D(3, strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.classifier = tf.keras.layers.Dense(1000, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.classifier(x)


def my_model_function():
    """
    Return an instance of MyModel.
    """
    model = MyModel()
    # Since no pretrained weights are available here,
    # weights remain randomly initialized.
    return model


def GetInput():
    """
    Return a random input tensor matching the expected input of MyModel.

    The original code was performing inference on ImageNet images.
    Inception v3 expects images typically of size 299x299 RGB (3 channels).
    Batch size is assumed 1 for this example.
    """
    return tf.random.uniform(shape=(1, 299, 299, 3), minval=0, maxval=1, dtype=tf.float32)


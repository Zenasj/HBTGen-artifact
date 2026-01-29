# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32) ← Input shape inferred from Stanford Dogs dataset preprocessing

import tensorflow as tf

IMG_LEN = 224
IMG_SHAPE = (IMG_LEN, IMG_LEN, 3)
N_BREEDS = 120

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Load MobileNetV2 base model with pretrained ImageNet weights, exclude the top classification head
        self.base_model = tf.keras.applications.MobileNetV2(
            input_shape=IMG_SHAPE,
            include_top=False,
            weights='imagenet'
        )
        self.base_model.trainable = False

        # Add pooling and classification head
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(N_BREEDS, activation='softmax')

    def call(self, inputs, training=False):
        """
        Forward pass.
        inputs: Tensor of shape (batch_size, 224, 224, 3), dtype float32 values scaled [0,1]
        returns: predictions (batch_size, N_BREEDS) softmax probabilities
        """
        x = self.base_model(inputs, training=training)
        x = self.global_pool(x)
        return self.classifier(x)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    """
    Returns a random tensor resembling a batch of images from Stanford Dogs dataset,
    shaped (batch_size, 224, 224, 3), values in [0,1], dtype float32.
    Batch size is chosen as 8 for example.
    """
    batch_size = 8
    # Use uniform random to simulate image data scaled as float32 in [0,1]
    return tf.random.uniform((batch_size, IMG_LEN, IMG_LEN, 3), dtype=tf.float32)


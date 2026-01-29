# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32) ← inferred input shape for the example model with IMAGE_SHAPE=(224,224,3)

import tensorflow as tf
import tensorflow_hub as hub

class MyModel(tf.keras.Model):
    def __init__(self, model_url="https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5", num_classes=10):
        super().__init__()
        self.feature_extractor = hub.KerasLayer(
            model_url,
            trainable=False,  # freeze pre-trained weights
            input_shape=(224, 224, 3),
            name="feature_extractor_layer"
        )
        self.classifier = tf.keras.layers.Dense(num_classes, activation="softmax", name="output_layer")

    def call(self, inputs, training=False):
        # The feature extractor expects input in [0,1] float range, with shape (B,224,224,3)
        # Ensure that the training argument passed to KerasLayer is False as per common practice for feature extractors
        features = self.feature_extractor(inputs, training=False)
        outputs = self.classifier(features)
        return outputs

def my_model_function():
    # Return an instance of MyModel with default parameters
    return MyModel()

def GetInput():
    # Return a random tensor input matching the expected shape and dtype
    # Input shape: batch size arbitrary, height=224, width=224, channels=3 (RGB image)
    # dtype float32, values in [0,1] simulating normalized image data
    # Batch size is chosen as 4 for example
    batch_size = 4
    input_tensor = tf.random.uniform((batch_size, 224, 224, 3), minval=0, maxval=1, dtype=tf.float32)
    return input_tensor


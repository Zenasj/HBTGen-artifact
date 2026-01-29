# tf.random.uniform((10, 64, 64, 3), dtype=tf.float32)  ← Input shape inferred as 10 images (shots) of 64x64 RGB

import tensorflow as tf
from tensorflow import keras

# This model simulates the use case described: 
# multiple images from fixed viewpoints (batch of 10 images) 
# and a decision based on these images (e.g., phone damage detection).
# It uses a Lambda layer with a safe serialization override as detailed in the issue,
# to avoid errors when saving & loading the model.

class SafeSaveLambda(keras.layers.Lambda):
    def _serialize_function_to_config(self, fn, allow_raw=False):
        # Override to fix serialization module name to "__main__" to ease loading across files
        output, output_type, module = super()._serialize_function_to_config(fn, allow_raw)
        return output, output_type, "__main__"

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple CNN layers to encode each image
        self.conv1 = keras.layers.Conv2D(16, 3, activation='relu', padding='same')
        self.pool1 = keras.layers.MaxPooling2D(2)
        self.conv2 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')
        self.pool2 = keras.layers.MaxPooling2D(2)
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(64, activation='relu')

        # Lambda layer using SafeSaveLambda subclass with simple processing function
        self.safe_lambda = SafeSaveLambda(lambda x: tf.reduce_mean(x, axis=[1, 2, 3]), 
                                          name="average_feature")

        # Final decision layer
        self.classifier = keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        # inputs shape assumed (batch=10, H=64, W=64, C=3)

        # Encode each image
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense(x)

        # Apply Lambda for global feature aggregation over batch dimension: 
        # according to use case, we process 10 shots jointly
        # The lambda reduces feature vector per image to scalar by averaging all channels,
        # but since input to lambda is shape (batch, features), axis chosen is 1 (features axis).
        # To reflect the original example, assume input is 10 images, but since this is the batch dimension,
        # we interpret the input batch as 10 separate images per sample.
        # Because Keras input batches are usually batch dimension, here we treat input shape as (10, H, W, C),
        # i.e. batch size 10 and single sample for the model call.

        # So we apply safe_lambda across the feature dimension:
        # after dense, x shape is (10, 64), reduce mean across axis=1 results in shape (10,)
        features_mean = self.safe_lambda(x)  # shape: (batch=10,)

        # Aggregate over the 10 feature means to a single scalar decision:
        decision_input = tf.reduce_mean(features_mean)  # scalar

        # Add batch dim for classifier (Keras Dense expects batch dim)
        decision_input = tf.expand_dims(decision_input, axis=0)

        out = self.classifier(decision_input)

        # Return scalar sigmoid output indicating phone damage probability
        return out

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Return a random tensor input matching the model input:
    # 10 images of shape 64x64 with 3 channels (RGB), float32
    return tf.random.uniform((10, 64, 64, 3), dtype=tf.float32)


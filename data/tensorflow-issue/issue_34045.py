# tf.random.uniform((3, 299, 299, 3), dtype=tf.int64) ← batch size 3, images 299x299x3, dtype according to dataset generation

import tensorflow as tf
import tensorflow_hub as hub

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.input_shape_ = [299, 299, 3]
        # Use the same feature extractor layer as original example from TF Hub
        self.feature_extractor = hub.KerasLayer(
            "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4",
            output_shape=[2048],
            input_shape=self.input_shape_,
            trainable=False
        )
        self.classifier = tf.keras.layers.Dense(units=6, activation='softmax')

    def call(self, inputs, training=False):
        # Forward pass through the feature extractor and classifier
        x = self.feature_extractor(inputs)
        output = self.classifier(x)
        return output

def my_model_function():
    """Construct and compile the model to mirror the Trainer's model."""
    model = MyModel()
    # Compile model as in original example
    model.compile(
        loss='categorical_crossentropy',
        optimizer='Adam',
        metrics=['accuracy']
    )
    return model

def GetInput():
    """Return a random valid input matching the expected input of MyModel."""
    # Batch size is 3 to reflect the example's batch size
    # The original dataset yields uint64 zeros but the generator in the example used int64 types.
    # For compatibility with the hub layer which expects float images,
    # we cast to float32 here and normalize values between 0 and 1 (common convention).
    batch_size = 3
    input_shape = (batch_size, 299, 299, 3)
    x = tf.random.uniform(input_shape, minval=0, maxval=255, dtype=tf.int64)
    x = tf.cast(x, tf.float32) / 255.0  # Normalize
    return x


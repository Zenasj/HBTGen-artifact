# tf.random.uniform((None, 32, 32, 128), dtype=tf.float32) ← inferred input shape from issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build a simple example model consistent with issue context:
        # Input shape: (None, 32, 32, 128)
        # Outputs: boxes (None, 4), scores (None,)
        # The model produces two outputs named distinctly to allow signature/name preservation.

        self.conv = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')
        self.pool = tf.keras.layers.GlobalAveragePooling2D()

        # Output heads
        # Boxes: 4 values (e.g., bounding box coordinates)
        # Scores: 1 value per batch example (e.g., confidence score)
        self.boxes_dense = tf.keras.layers.Dense(4, name="boxes_dense")
        self.scores_dense = tf.keras.layers.Dense(1, name="scores_dense")
        # Identity layers to preserve output names as suggested in the issue
        self.boxes_identity = tf.keras.layers.Lambda(lambda x: tf.identity(x), name="boxes")
        self.scores_identity = tf.keras.layers.Lambda(lambda x: tf.squeeze(tf.identity(x), axis=-1), name="scores")

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.pool(x)
        boxes = self.boxes_dense(x)
        scores = self.scores_dense(x)
        # Wrap outputs with identity layers to preserve output tensor names for TFLite SignatureDef
        boxes_named = self.boxes_identity(boxes)
        scores_named = self.scores_identity(scores)
        return {"boxes": boxes_named, "scores": scores_named}


def my_model_function():
    # Create an instance of MyModel.
    model = MyModel()
    # Build the model by passing a dummy input (to initialize weights)
    dummy_input = GetInput()
    # Run once to build variables
    _ = model(dummy_input)
    return model


def GetInput():
    # Return a random tensor input matching expected shape
    # Shape: (batch_size, height, width, channels)
    # Using batch size 1 for simplicity.
    return tf.random.uniform((1, 32, 32, 128), dtype=tf.float32)


# tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.input_shape_ = (224, 224, 3)

        # Base MobileNetV2 feature extractor without top layers, pretrained on imagenet.
        # This block is inference only with pretrained weights.
        self.base_model = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=self.input_shape_)

        # Pooling & classification head consistent with the functional/sequential examples in the issue.
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(256, activation="relu", name="descriptor")
        self.dense2 = tf.keras.layers.Dense(2, activation="softmax", name="probs")

    def call(self, inputs, training=False):
        """
        Forward pass.

        This matches the functional model described in the issue:
            Inputs (batch, 224, 224, 3)
            -> MobileNetV2 base (no top)
            -> GlobalAveragePooling2D
            -> Dense(256, relu)
            -> Dense(2, softmax)

        Compatible with tf2.20.0 and XLA compilation.
        """
        x = self.base_model(inputs, training=training)
        x = self.global_pool(x)
        x = self.dense1(x)
        outputs = self.dense2(x)
        return outputs


def my_model_function():
    """
    Returns a new instance of the MyModel class
    """
    return MyModel()


def GetInput():
    """
    Returns a single batch input tensor matching the expected input shape (1, 224, 224, 3)
    with values sampled uniformly in [0, 1).

    This input tensor is compatible with the MyModel call method.
    """
    # Batch size of 1 for example input
    batch_size = 1
    input_shape = (batch_size, 224, 224, 3)
    return tf.random.uniform(input_shape, dtype=tf.float32)


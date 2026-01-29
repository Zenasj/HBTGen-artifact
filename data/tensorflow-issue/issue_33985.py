# tf.random.uniform((1, 224, 224, 3), dtype=tf.float32) ← inferred input shape from MobileNetV2 input_shape

import tensorflow as tf

class L2NormalizeLayer(tf.keras.layers.Layer):
    def __init__(self, name="normalize", **kwargs):
        super(L2NormalizeLayer, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        # axis=1 following original logic, which shapes to (batch, features)
        return tf.keras.backend.l2_normalize(inputs, axis=1)

    def get_config(self):
        config = super(L2NormalizeLayer, self).get_config()
        return config


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Base pretrained feature extractor without top layers
        self.base_model = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=(224, 224, 3)
        )
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.dense256 = tf.keras.layers.Dense(256, activation="relu")
        self.l2norm = L2NormalizeLayer(name="embeddings")
        self.output_dense = tf.keras.layers.Dense(2, activation="softmax", name="probs")

    def call(self, inputs, training=None):
        x = self.base_model(inputs, training=training)  # (B, 7, 7, 1280) approx
        x = self.gap(x)                                 # (B, 1280)
        x = self.dense256(x)                            # (B, 256)
        embeddings = self.l2norm(x)                     # normalized embeddings (B, 256)
        probs = self.output_dense(x)                     # (B, 2)
        # Because original issue focused on accessing both outputs,
        # here we expose them both as tuple outputs.
        return probs, embeddings


def my_model_function():
    # Return an instance of MyModel, ready for use
    return MyModel()

def GetInput():
    # Return random input tensor matching input expected by MyModel (batch size 1)
    return tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)


# tf.random.uniform((1, 224, 224, 3), dtype=tf.float32) ← inferred input shape to match MobileNetV2 input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Base MobileNetV2 without the top layers, imagenet weights, input shape 224x224x3
        self.base_model = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=(224, 224, 3)
        )
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense_embeddings = tf.keras.layers.Dense(256, activation="relu", name="embeddings")
        self.dense_probs = tf.keras.layers.Dense(2, activation="softmax", name="probs")

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.global_pool(x)
        embeddings = self.dense_embeddings(x)
        probs = self.dense_probs(embeddings)
        # Output as tuple of embeddings and probs, as in original code getting both outputs
        return {"probs": probs, "embeddings": embeddings}

def my_model_function():
    # Return instance of the model, weights loaded from imagenet internally for base_model
    # No further initialization needed as base_model includes pretrained weights
    return MyModel()

def GetInput():
    # Return a sample input tensor matching the model input shape: batch size 1, 224x224x3 RGB image
    return tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)


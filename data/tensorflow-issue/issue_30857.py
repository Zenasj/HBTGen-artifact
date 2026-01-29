# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32) ← batch size B is dynamic, images are 224x224 RGB

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Base feature extractor: ResNet50 (no weights, no top)
        self.base_model = tf.keras.applications.ResNet50(
            input_shape=(224, 224, 3), include_top=False, weights=None)
        
        # Global max pooling to reduce spatial dimensions
        self.pool = tf.keras.layers.GlobalMaxPooling2D()
        
        # Dropout for regularization
        self.dropout = tf.keras.layers.Dropout(0.4)
        
        # Final classification layer with 9 classes, softmax activation
        self.pred = tf.keras.layers.Dense(9, activation='softmax')

    def call(self, inputs, training=False):
        # inputs: image tensor with shape [B, 224, 224, 3], dtype float32 assumed
        x = self.base_model(inputs, training=training)
        x = self.pool(x)
        x = self.dropout(x, training=training)
        x = self.pred(x)
        return x


def my_model_function():
    # Return a fresh instance of MyModel
    return MyModel()


def GetInput():
    # Return a random batch input tensor matching the input expected by MyModel
    # Here we choose a batch size of 32 to be consistent with the example
    return tf.random.uniform(shape=(32, 224, 224, 3), dtype=tf.float32)


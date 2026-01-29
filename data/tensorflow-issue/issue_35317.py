# tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Use pretrained ResNet50 as backbone model with input shape (224,224,3)
        # No top layers since this was a feature extraction example; here, keep default include_top=True but can be changed.
        self.backbone = tf.keras.applications.ResNet50(input_shape=(224, 224, 3), include_top=True)

    def call(self, inputs, training=False):
        # Forward pass through ResNet50 model
        return self.backbone(inputs, training=training)

def my_model_function():
    # Return an instance of MyModel with ResNet50 initialized
    return MyModel()

def GetInput():
    # Return a random tensor with shape (1, 224, 224, 3) matching expected input of MyModel
    # dtype tf.float32 in [0, 1] range consistent with preprocessing in original generator
    return tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)


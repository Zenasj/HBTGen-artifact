# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32) ← input shape inferred from model input_shape=(224, 224, 3)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original model was a Sequential with:
        # GlobalMaxPool2D(input_shape=(224,224,3)) followed by Dense(1000, activation='softmax')
        self.pool = tf.keras.layers.GlobalMaxPool2D()
        self.classifier = tf.keras.layers.Dense(1000, activation="softmax")
    
    def call(self, inputs, training=False):
        # Forward pass as in original model
        x = self.pool(inputs)
        x = self.classifier(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor matching the input expected by MyModel
    # Batch size B chosen as 1024 in original issue, but for compatibility here we pick a reasonable batch size
    # Note: The original dataset batch size was 1024, so we choose B=32 here for example input
    B = 32
    H, W, C = 224, 224, 3
    # Use float32 as model input is cast to float32 during preprocessing
    return tf.random.uniform(shape=(B, H, W, C), dtype=tf.float32)


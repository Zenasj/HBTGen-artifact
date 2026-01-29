# tf.random.uniform((GLOBAL_BATCH_SIZE, 224, 224, 3), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

class MyModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        # Using ResNet50 base as in the original code snippet
        self.base_model = tf.keras.applications.ResNet50(
            input_shape=(224, 224, 3),
            weights='imagenet',
            include_top=False)
        self.pool = GlobalAveragePooling2D()
        self.classifier = Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=training)
        x = self.pool(x)
        x = self.classifier(x)
        return x

def my_model_function():
    # Example number of classes - this should be replaced to actual dataset classes at runtime
    num_classes = 5  # assumption: dataset used has 5 classes (like flower classification)
    model = MyModel(num_classes=num_classes)
    # Initialize the model weights by running with dummy input once
    dummy_input = GetInput()
    _ = model(dummy_input, training=False)
    return model

def GetInput():
    # Generate dummy input tensor matching expected input shape for MyModel
    # Batch size chosen as 32 for example; can be adjusted as needed
    BATCH_SIZE = 32
    H, W, C = 224, 224, 3
    # tf.random.uniform simulates normalized images in range [0,1]
    return tf.random.uniform((BATCH_SIZE, H, W, C), dtype=tf.float32)


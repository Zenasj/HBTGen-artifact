# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32)  ← inferred input shape based on ResNet50 input_shape=(224,224,3)

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import models, layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Base ResNet50 model without top layers, imagenet weights
        self.base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(10, activation='sigmoid')  # multi-label 10 classes
    
        # Use F1Score metric with average='macro' (to avoid multi-label per-class array issues)
        # This matches the recommended fix for multi-label use case in the issue discussion.
        self.f1_score_metric = tf.keras.metrics.F1Score(
            num_classes=10,  # 10 classes
            threshold=0.5,
            average='macro'  # Important: prevents returning array per class which causes ModelCheckpoint issue
        )

    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=training)
        x = self.global_pool(x)
        out = self.classifier(x)
        return out

    def compute_f1(self, y_true, y_pred):
        # Utility to update and get F1 score metric for monitoring
        self.f1_score_metric.update_state(y_true, y_pred)
        return self.f1_score_metric.result()
    
def my_model_function():
    # Instantiate the model and compile it with correct metric to avoid checkpoint error
    model = MyModel()
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
        # Using F1Score with average='macro' to avoid "ValueError: ambiguous array" in checkpoint
        metrics=[tf.keras.metrics.F1Score(num_classes=10, threshold=0.5, average='macro')]
    )
    return model

def GetInput():
    # Generate a dummy batch of inputs to match (B, 224, 224, 3)
    # Using batch size 4 as a reasonable default
    batch_size = 4
    input_tensor = tf.random.uniform((batch_size, 224, 224, 3), dtype=tf.float32)
    return input_tensor


# tf.random.uniform((1, 128, 128, 3), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Using EfficientNetB0 as base with frozen layers
        self.base_model = EfficientNetB0(include_top=False, input_shape=(128, 128, 3))
        for layer in self.base_model.layers:
            layer.trainable = False
        
        # Additional top layers after base model
        self.conv = layers.Conv2D(64, (3, 3), activation='relu')
        self.maxpool = layers.MaxPooling2D((2, 2))
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.out_dense = layers.Dense(6, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=training)
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.global_avg_pool(x)
        out = self.out_dense(x)
        return out

def my_model_function():
    """
    Returns an instance of MyModel, compiled with:
    - CategoricalCrossentropy loss
    - Adam optimizer
    - A placeholder metric 'kappa_score' since original is not defined here.
    """
    model = MyModel()

    # Define placeholder kappa_score metric:
    # Since kappa_score is not defined in the issue,
    # using categorical_accuracy as a stand-in for demonstration.
    def kappa_score(y_true, y_pred):
        # Placeholder metric logic (should be replaced with real implementation)
        return tf.keras.metrics.categorical_accuracy(y_true, y_pred)
    
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=Adam(),
        metrics=[kappa_score]
    )
    return model

def GetInput():
    """
    Returns a random input tensor suitable as input to MyModel:
    - Batch size 1
    - Shape (128, 128, 3)
    - dtype float32 (typical for images normalized to [0,1])
    """
    return tf.random.uniform((1, 128, 128, 3), dtype=tf.float32)


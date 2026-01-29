# tf.random.uniform((B, 28, 28, 3), dtype=tf.float32) ← Inputs are pairs of 28x28 RGB images

import tensorflow as tf
import numpy as np

def euclidean_distance(vects):
    """
    Compute Euclidean distance between two tensors of shape (B, D).
    Returns shape (B, 1).
    """
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def contrastive_loss(margin=1):
    """
    Contrastive loss function with margin.
    y_true indicates whether pairs are similar (1) or dissimilar (0).
    y_pred is Euclidean distance between feature vectors.
    """
    def loss_fn(y_true, y_pred):
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - y_pred, 0))
        return tf.math.reduce_mean((1 - y_true) * square_pred + (y_true) * margin_square)
    return loss_fn

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Shared CNN subnetwork
        self.cnn = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (2, 2), activation="tanh", padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv2D(64, (2, 2), activation="tanh", padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv2D(128, (2, 2), activation="tanh", padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv2D(256, (2, 2), activation="tanh", padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation='tanh')
        ])
        # BatchNorm layer applied to distance
        self.batch_norm = tf.keras.layers.BatchNormalization()
        # Final classification layer
        self.classifier = tf.keras.layers.Dense(1, activation='sigmoid')

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        """
        inputs: list or tuple of two tensors, each shape (B, 28, 28, 3)
        Process each input image through shared CNN,
        compute distance, normalize and classify similarity.
        """
        img_A, img_B = inputs
        
        # Extract feature vectors (B, 64)
        feature_vector_A = self.cnn(img_A, training=training)
        feature_vector_B = self.cnn(img_B, training=training)
        
        # Compute Euclidean distance (B, 1)
        distance = euclidean_distance([feature_vector_A, feature_vector_B])
        
        # Normalize distance
        norm_dist = self.batch_norm(distance, training=training)
        
        # Predict similarity score (B, 1)
        out = self.classifier(norm_dist)
        return out

def my_model_function():
    """
    Returns an instance of MyModel.
    """
    return MyModel()

def GetInput():
    """
    Generate a random batch of image pairs as input to MyModel.
    Shape: (batch_size, 28, 28, 3) for each image in the pair.
    We'll use uint8 values scaled to float32 [0, 1] to simulate RGB images.
    """
    batch_size = 32
    # Random floats simulating images, shape (batch_size, 28, 28, 3)
    img_A = tf.random.uniform((batch_size, 28, 28, 3), minval=0, maxval=255, dtype=tf.float32) / 255.0
    img_B = tf.random.uniform((batch_size, 28, 28, 3), minval=0, maxval=255, dtype=tf.float32) / 255.0
    return [img_A, img_B]

# Notes on assumptions and reconstruction:
# - Inputs are pairs of 28x28 images expanded to 3 channels (converted grayscale to RGB).
# - The model uses the CNN subnetwork to encode each image, then calculates Euclidean distance.
# - BatchNormalization and a sigmoid Dense layer output similarity prediction.
# - The loss function is contrastive loss as provided.
# - The forward call is decorated with @tf.function(jit_compile=True) for XLA compatibility.
# - GetInput() creates a batch of random images compatible with model input.


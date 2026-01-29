# tf.random.uniform((B, 10), dtype=tf.float32) ← Input shape inferred from Input(shape=(10,))

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import kullback_leibler_divergence

class MyModel(tf.keras.Model):
    """
    A model with a custom clustering layer.

    The ClusteringLayer implements a clustering assignment based on Student's t-distribution kernel.

    Notes:
    - Fixed issues from older TF version with input shape handling by using tf.TensorShape properly.
    - ClusteringLayer's weight matrix W has shape (output_dim, input_dim).
    - The call method computes soft assignments q based on distance to cluster centers.
    """

    def __init__(self, output_dim=5, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.alpha = alpha
        # The custom clustering layer as a submodule
        self.clustering_layer = ClusteringLayer(self.output_dim, self.alpha)

    def call(self, inputs, training=None):
        # Forward pass returns clustering assignments q
        return self.clustering_layer(inputs)

def clustering_loss(y_true, y_pred):
    """
    Custom clustering loss based on KL divergence between target distribution p and predicted q.

    p is calculated as normalized square of q as per:
    p_ij = (q_ij^2 / sum_i q_ij) / normalization
    """
    # Square of predicted q
    a = K.square(y_pred) / K.sum(y_pred, axis=0, keepdims=True)
    p = a / K.sum(a, axis=1, keepdims=True)
    # Compute KL divergence between p and q
    loss = kullback_leibler_divergence(p, y_pred)
    return loss

class ClusteringLayer(Layer):
    """
    Custom Keras layer that computes soft cluster assignments based on Student's t-distribution kernel.

    Attributes:
        output_dim: Number of clusters (int)
        alpha: Parameter in Student's t-distribution (float)
    """

    def __init__(self, output_dim, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.alpha = alpha
        self.W = None  # Cluster centroids weights

    def build(self, input_shape):
        # input_shape is a TensorShape, typically (batch_size, input_dim)
        input_dim = input_shape[-1]
        # W shape: (output_dim, input_dim) representing cluster centers
        self.W = self.add_weight(
            name='kernel',
            shape=(self.output_dim, input_dim),
            initializer=tf.keras.initializers.Identity(),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # Expand dims of inputs to (batch_size, 1, input_dim)
        expanded_inputs = tf.expand_dims(inputs, axis=1)  
        # Compute squared euclidean distance between inputs and cluster centers:
        # shape (batch_size, output_dim)
        distance = tf.reduce_sum(tf.square(expanded_inputs - self.W), axis=2)
        # Student's t-distribution kernel (unnormalized)
        numerator = 1.0 / (1.0 + (distance / self.alpha))
        # Raise to power as per formula
        q = numerator ** ((self.alpha + 1.0) / 2.0)
        # Normalize over clusters (dim=1)
        q_sum = tf.reduce_sum(q, axis=1, keepdims=True)
        q_normalized = q / q_sum
        return q_normalized

    def compute_output_shape(self, input_shape):
        # Output shape is (batch_size, output_dim)
        return tf.TensorShape((input_shape[0], self.output_dim))

def my_model_function():
    # Instantiate MyModel with clustering layer of output_dim=5 and default alpha=1
    return MyModel(output_dim=5, alpha=1.0)

def GetInput():
    # Return a random input tensor of shape (batch_size=1, 10) with dtype float32 matching model input
    return tf.random.uniform((1, 10), dtype=tf.float32)


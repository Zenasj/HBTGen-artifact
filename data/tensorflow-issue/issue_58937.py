# tf.random.uniform((10, 100), dtype=tf.float16) ← Input is batch of 10 vectors, 100-d features, dtype float16

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=100):
        super().__init__()
        self.num_classes = num_classes
        # Weights are variables added in build()
        self._weights = None

    def build(self, input_shape):
        # Create a weight matrix of shape (feature_dim, num_classes)
        self._weights = self.add_weight(
            name="W",
            shape=(input_shape[-1], self.num_classes),
            initializer="glorot_normal",
            trainable=True,
            dtype=self.dtype  # Should track the model dtype policy
        )
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_classes)

    @tf.function  # As per original code, but note issue described with save caused by this decorator
    def call(self, inputs: tf.Tensor):
        """
        Forward pass: compute cosine similarity logits.

        Assumptions:
        - Inputs: float16 tensor of shape (batch_size, feature_dim)
        - Weights: stored as AutoCastVariable, dtype float32 internally but autocast to float16 during call.
        - Output: logits computed by (embedding_normalized @ weights_normalized),
          both normalized along feature axis.
        """
        # Normalize input embeddings along last axis (features)
        embedding_normalized = tf.nn.l2_normalize(inputs, axis=1)

        # Normalize weights (centroids) along last axis
        weights_normalized = tf.nn.l2_normalize(self._weights, axis=0)  # Normalize weights column-wise
        # Correction: Original code normalized axis=1 for weights of shape (features, classes),
        # which would normalize rows (features), but typical cosine similarity expects weights normalized per class vector,
        # so axis=0 over dim features to get per-class centroids of norm 1.

        # Compute cosine similarity logits: (batch_size, features) @ (features, num_classes) -> (batch_size, num_classes)
        logits = tf.matmul(embedding_normalized, weights_normalized)

        return logits

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
        })
        return config

def my_model_function():
    """
    Returns an instance of MyModel set to mixed precision policy 'mixed_float16' as in original snippet,
    so that weights are AutoCastVariables and inputs are expected float16.
    """
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    model = MyModel(num_classes=100)
    # Model will be built on first call()
    return model

def GetInput():
    """
    Return a random tensor input that matches MyModel input:
    shape (batch=10, features=100), dtype float16 as per mixed precision policy in example.
    """
    return tf.random.uniform(shape=(10, 100), dtype=tf.float16)


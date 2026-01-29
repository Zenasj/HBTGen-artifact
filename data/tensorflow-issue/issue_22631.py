# tf.random.uniform((1, 128, 128, 3), dtype=tf.float32)
import tensorflow as tf
import numpy as np

# Discriminative loss attempting to compute cluster means per image in batch
def discriminative_loss(y_true, y_pred):
    """Computes loss for a batch of images.

    Args:
        y_true: Tensor of shape (batch_size, height, width, 1), containing ground truth instance ids.
        y_pred: Tensor of shape (batch_size, height, width, d), d-dimensional embedding vectors per pixel.

    Returns:
        Tensor of shape (batch_size,) with the loss for each image in the batch.
    """

    def compute_loss(inputs):
        label = inputs[0]   # shape (H, W, 1)
        prediction = inputs[1]  # shape (H, W, d)
        
        # Find unique clusters (instance ids) in label, flattening label
        clusters, _ = tf.unique(tf.reshape(label, [-1]))
        
        # For each cluster, compute cluster mean vector of corresponding pixels
        def compute_mean(c):
            mask = tf.equal(label[:, :, 0], c)  # boolean mask shape (H, W)
            masked_pixels = tf.boolean_mask(prediction, mask)  # shape (num_pixels_in_cluster, d)
            cluster_mean = tf.reduce_mean(masked_pixels, axis=0)
            return cluster_mean
        
        cluster_means = tf.map_fn(compute_mean, clusters, dtype=tf.float32)
        
        # A dummy reduction over cluster means (placeholder for real loss)
        return tf.reduce_mean(cluster_means)

    # Compute loss for each image in batch using tf.map_fn
    losses = tf.map_fn(compute_loss, (y_true, y_pred), dtype=tf.float32)
    return losses

def discriminative_loss_working(y_true, y_pred):
    # Loss computed only for first image in batch (for debugging / fallback)
    prediction = y_pred[0]
    label = y_true[0]
    clusters, _ = tf.unique(tf.reshape(label, [-1]))
    def compute_mean(c):
        mask = tf.equal(label[:, :, 0], c)
        masked_pixels = tf.boolean_mask(prediction, mask)
        cluster_mean = tf.reduce_mean(masked_pixels, axis=0)
        return cluster_mean
    cluster_means = tf.map_fn(compute_mean, clusters, dtype=tf.float32)
    return tf.reduce_mean(cluster_means)


class MyModel(tf.keras.Model):
    def __init__(self, input_shape):
        super(MyModel, self).__init__()
        # Single 2D conv layer with 4 filters and 1x1 kernel (simple feature extractor)
        self.conv = tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 1))
        
    def call(self, inputs):
        # forward pass: apply conv layer
        return self.conv(inputs)


def my_model_function():
    # Return an instance of MyModel with input shape fixed as (1,128,128,3)
    return MyModel(input_shape=(1, 128, 128, 3))


def GetInput():
    # Return a random tensor input matching MyModel's expected input shape:
    # batch size 1, height 128, width 128, channels 3, dtype float32
    return tf.random.uniform((1, 128, 128, 3), dtype=tf.float32)


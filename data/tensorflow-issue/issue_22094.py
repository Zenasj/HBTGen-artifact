# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Input shape inferred as (batch_size, 128, 128, 3) with batch_size=1 for example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple conv layer as shown in the provided code snippet
        self.conv = tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 1))

    def call(self, input):
        # Forward pass through conv layer
        return self.conv(input)

def discriminative_loss(y_true, y_pred):
    """
    Computes loss for a batch of images.

    Args:
        y_true: Tensor of shape (n, h, w, 1), where each element contains the ground truth instance id.
        y_pred: Tensor of shape (n, h, w, d), d-dimensional vector for each pixel for each image in the batch.

    Returns:
        Tensor of shape (n,) representing loss per image.
    """
    # Note: y_true shape includes a last dimension (1), consistent with the input generator's output.

    def compute_loss(input):
        label = input[0]  # shape (h, w, 1)
        prediction = input[1]  # shape (h, w, d)

        # Find unique cluster ids in the label flattened
        clusters, _ = tf.unique(tf.reshape(label, [-1]))

        def compute_mean(c):
            # Create mask for pixels belonging to cluster c
            mask = tf.equal(label[:, :, 0], c)  # shape (h, w), bool mask

            # boolean_mask extracts pixels belonging to this cluster from prediction
            masked_pixels = tf.boolean_mask(prediction, mask)
            # safe reduce_mean: handle case if masked_pixels empty by tf.reduce_mean, though it should not be empty ideally
            cluster_mean = tf.reduce_mean(masked_pixels, axis=0)
            return cluster_mean

        # Compute cluster means for all clusters
        cluster_means = tf.map_fn(compute_mean, clusters, dtype=tf.float32)
        # Loss is mean over these cluster means (per cluster feature vector mean)
        return tf.reduce_mean(cluster_means)

    # Map compute_loss over batch dimension
    # Inputs to map_fn: tuple (y_true, y_pred) unpacked element-wise => input = (label_i, pred_i)
    losses = tf.map_fn(compute_loss, (y_true, y_pred), dtype=tf.float32)
    return losses

def my_model_function():
    # Return an instance of MyModel with no required external init parameters
    return MyModel()

def GetInput():
    # Returns a tuple (y_true, y_pred) matching expected input to discriminative_loss inside MyModel training
    
    # Using batch size 1, height and width 128, depth 3 channels input for model
    batch_size = 1
    height = 128
    width = 128
    d = 4  # feature dimension from model output conv filters count

    # y_true shape (batch_size, height, width, 1): instance ids in range 11000-11015 as in example
    y_true = tf.random.uniform(
        shape=(batch_size, height, width, 1),
        minval=11000,
        maxval=11015,
        dtype=tf.int32
    )
    y_true = tf.cast(y_true, tf.float32)  # Convert to float32 since example shows float32 labels (though IDs are int)

    # y_pred shape (batch_size, height, width, d): model output features (random here)
    y_pred = tf.random.uniform(
        shape=(batch_size, height, width, d),
        dtype=tf.float32
    )
    return (y_true, y_pred)


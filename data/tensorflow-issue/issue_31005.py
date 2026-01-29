# tf.random.uniform((1, 128, 128, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, input_shape):
        super(MyModel, self).__init__()
        # A simple Conv2D layer as per the issue example
        self.conv = tf.keras.layers.Conv2D(filters=4, kernel_size=(1,1), padding='same')

    def call(self, inputs):
        # Forward pass through the convolution layer
        return self.conv(inputs)

def discriminative_loss(y_true, y_pred):
    """
    Computes a discriminative loss for a batch of images.
    Args:
        y_true: tensor of shape (n, h, w, 1), where each element contains the ground truth instance id.
        y_pred: tensor of shape (n, h, w, d), d-dimensional vector embeddings for each pixel per image.
    Returns:
        loss: tensor of shape (n,), representing the loss per image in batch.
    
    Notes:
    - This implementation mimics the issue's original logic where the loss is computed
      by first computing the mean embedding vector per cluster (instance) and then 
      reducing these means to a loss value. Here, for demonstration, the reduced mean
      of cluster means is returned as a scalar loss per image.
    """
    def compute_loss(inputs):
        label = inputs[0]  # shape (h, w, 1)
        prediction = inputs[1]  # shape (h, w, d)

        # Flatten the label to find unique clusters (instance ids)
        clusters, _ = tf.unique(tf.reshape(label, [-1]))

        def compute_mean(c):
            # Create a mask for pixels belonging to cluster c; label shape (h,w,1)
            # Squeeze last dim for mask comparison
            mask = tf.equal(tf.squeeze(label, axis=-1), c)

            # Apply mask to prediction pixels, shape (h,w,d)
            masked_pixels = tf.boolean_mask(prediction, mask)

            # Compute mean embedding vector of the cluster
            cluster_mean = tf.reduce_mean(masked_pixels, axis=0)
            return cluster_mean

        # Compute the mean embedding vector for each cluster (instance)
        cluster_means = tf.map_fn(compute_mean, clusters, fn_output_signature=tf.float32)

        # For demonstration, return mean of cluster means as loss for this image
        # (in practice, loss would be more complex)
        return tf.reduce_mean(cluster_means)

    # Use tf.map_fn to compute loss per image in the batch
    losses = tf.map_fn(compute_loss, (y_true, y_pred), fn_output_signature=tf.float32)
    return losses

def my_model_function():
    # Input shape fixed to (1, 128, 128, 3) based on the issue example
    input_shape = (1, 128, 128, 3)
    return MyModel(input_shape)

def GetInput():
    # Returns a random float tensor matching the model input shape: (1,128,128,3)
    # Using tf.random.uniform for compatibility with tf.float32 input dtype
    return tf.random.uniform(shape=(1, 128, 128, 3), dtype=tf.float32)


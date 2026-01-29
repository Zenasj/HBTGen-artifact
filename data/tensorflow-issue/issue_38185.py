# tf.random.uniform((1024, 1000), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No parameters needed since this model acts as a deterministic wrapper
        # around softmax_cross_entropy and sparse_softmax_cross_entropy for testing.

    def call(self, inputs, training=False):
        """
        Expects inputs as a tuple: (logits, labels, exclusive_labels_flag)
        
        - logits: [batch_size, classes_count], tf.float32
        - labels: either sparse labels [batch_size] (int32) if exclusive_labels_flag is True,
                  or distribution labels [batch_size, classes_count] (float32) if False
        - exclusive_labels_flag: bool scalar True if labels are sparse, False if distribution
        
        Returns:
        - loss: Tensor of shape [batch_size] with the per-example loss output
        """

        logits, labels, exclusive_labels = inputs

        # Determine whether to use sparse or categorical crossentropy in a deterministic way
        if exclusive_labels:
            # For sparse labels, use sparse_categorical_crossentropy (from_logits=False)
            # Need to first compute softmax to mimic deterministic path
            probs = tf.nn.softmax(logits)
            # sparse_categorical_crossentropy expects labels as int32 and probs not logits
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                labels, probs, from_logits=False)
        else:
            # For distribution labels, use categorical_crossentropy (from_logits=False)
            probs = tf.nn.softmax(logits)
            loss = tf.keras.losses.categorical_crossentropy(
                labels, probs, from_logits=False)

        return loss


def my_model_function():
    # Return an instance of MyModel, no special initialization needed
    return MyModel()


def GetInput():
    """
    Returns a tuple (logits, labels, exclusive_labels_flag) that matches the input expected by MyModel.

    - logits: float32 tensor with shape [1024, 1000]
    - labels: either int32 tensor with shape [1024] (for sparse labels) OR
              float32 tensor with shape [1024, 1000] (for distribution labels)
    - exclusive_labels_flag: boolean tensor scalar indicating label type
    """
    batch_size = 1024
    classes_count = 1000

    # Create random logits: uniform random floats between -1 and 1
    logits = tf.random.uniform((batch_size, classes_count), minval=-1, maxval=1,
                               dtype=tf.float32)

    # Randomly choose to generate sparse or distribution labels
    # For deterministic demonstration, just pick sparse labels here; user can swap
    exclusive_labels_flag = True

    if exclusive_labels_flag:
        # sparse labels: integer class indices between [0, classes_count)
        labels = tf.random.uniform((batch_size,), minval=0, maxval=classes_count,
                                   dtype=tf.int32)
    else:
        # distribution labels: floats normalized to sum to 1 along classes axis
        raw = tf.random.uniform((batch_size, classes_count), minval=0, maxval=1,
                                dtype=tf.float32)
        labels = raw / tf.reduce_sum(raw, axis=1, keepdims=True)

    return (logits, labels, exclusive_labels_flag)


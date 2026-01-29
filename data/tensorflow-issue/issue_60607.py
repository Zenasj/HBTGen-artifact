# tf.random.uniform((B, seq_len), dtype=tf.int32)
import tensorflow as tf

NUM_CLASSES = 104

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Load the LaBSE model from TF Hub as a KerasLayer
        # NOTE: In an actual environment this requires internet access.
        # Here we assume the presence of the hub.KerasLayer wrapper.
        # For demonstration, we'll add a placeholder layer if hub is unavailable.
        try:
            import tensorflow_hub as hub
            self.encoder = hub.KerasLayer(
                "https://tfhub.dev/google/LaBSE/2",
                trainable=True,
                signature="default",
                output_key="pooled_output",
            )
        except ImportError:
            # Placeholder: Just use an embedding + global average pool for demo
            self.encoder = tf.keras.Sequential([
                tf.keras.layers.Embedding(32000, 768),
                tf.keras.layers.GlobalAveragePooling1D(),
            ])

        # Two output heads for multi-label classification (events and return_events)
        self.events = tf.keras.layers.Dense(NUM_CLASSES, activation="sigmoid", name="events")
        self.return_events = tf.keras.layers.Dense(NUM_CLASSES, activation="sigmoid", name="return_events")

    def call(self, inputs, training=False):
        """
        inputs: dict with keys
            - input_word_ids: tf.int32 tensor of shape (B, seq_len)
            - input_mask: tf.int32 tensor of shape (B, seq_len)
            - input_type_ids: tf.int32 tensor of shape (B, seq_len)
        returns tuple of two tensors:
            (events_output, return_events_output), both shape (B, NUM_CLASSES)
        """
        # The LaBSE model expects dict input with those keys
        encoding = self.encoder({
            "input_word_ids": inputs["input_word_ids"],
            "input_mask": inputs["input_mask"],
            "input_type_ids": inputs["input_type_ids"],
        })  # shape (B, embedding_dim), typically 768

        events_out = self.events(encoding)
        return_events_out = self.return_events(encoding)
        return events_out, return_events_out

def macro_double_soft_f1(y, y_hat):
    """
    Compute the macro soft F1-score cost (average 1 - soft-F1 across all labels).
    Uses probability values and accounts for positive and negative classes.

    Args:
        y: int32 Tensor of shape (B, NUM_CLASSES) - target multi-labels (0 or 1)
        y_hat: float32 Tensor of shape (B, NUM_CLASSES) - predicted probabilities

    Returns:
        scalar Tensor: averaged cost (lower is better)
    """
    y = tf.cast(y, tf.bfloat16)
    y_hat = tf.cast(y_hat, tf.bfloat16)

    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)

    soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)

    cost_class1 = 1 - soft_f1_class1
    cost_class0 = 1 - soft_f1_class0
    cost = 0.5 * cost_class1 + cost_class0

    macro_cost = tf.reduce_mean(cost)
    return macro_cost

def my_model_function():
    """
    Returns an instance of MyModel. No weights loaded here,
    but in actual usage weights can be loaded externally.
    """
    return MyModel()

def GetInput():
    """
    Returns an example input dictionary compatible with MyModel's call().
    Shapes and types based on the minimal example from the issue:
    batch size 2, sequence length 7.
    """
    batch_size = 2
    seq_len = 7

    # Use random int32 token IDs in range [0, 32000) as a placeholder vocab size
    input_word_ids = tf.random.uniform((batch_size, seq_len), minval=0, maxval=32000, dtype=tf.int32)
    # Mask is 0 or 1; let's randomly assign mostly ones with a few zeros
    input_mask = tf.ones((batch_size, seq_len), dtype=tf.int32)
    input_type_ids = tf.zeros((batch_size, seq_len), dtype=tf.int32)

    return {
        "input_word_ids": input_word_ids,
        "input_mask": input_mask,
        "input_type_ids": input_type_ids,
    }


# tf.random.uniform((B, 256), dtype=tf.int32) ← input shape corresponds to (batch_size, max_seq_length=256)

import tensorflow as tf

# This code is adapted and modernized from the original TensorFlow 1.x style code for a BERT layer setup,
# simplified and corrected assumptions to work in TensorFlow 2.x eager and keras subclassing model.

# We provide a custom BERT layer that wraps TensorFlow Hub BERT module and exposes fine-tuning options.
# The model expects three inputs: input_ids, input_mask, and segment_ids with shape (batch_size, 256).

# Assumptions:
# - Pooling is done with "first" token pooling by default.
# - BERT Hub module is loaded directly (the original used hub.Module which is TF1; here we simulate interface).
# - For TF2.20 compatibility, we mock a minimal TF Hub layer, since the TF1 hub.Module is deprecated.
# - The final output is a 5-class sigmoid classification.

# Note: In realistic usage, users should load from pretrained BERT TF2 Hub layers
# or from Huggingface transformers for better support.

class BertLayer(tf.keras.layers.Layer):
    def __init__(self, n_fine_tune_layers=3, pooling="first", bert_path=None, **kwargs):
        super().__init__(**kwargs)
        self.n_fine_tune_layers = n_fine_tune_layers
        self.pooling = pooling
        self.output_size = 768
        self.bert_path = bert_path

        if pooling not in ["first", "mean"]:
            raise ValueError(f"Pooling must be 'first' or 'mean', got {pooling}")

        # Instead of TF1 hub.Module, in TF2 we use hub.KerasLayer or a placeholder for demonstration.
        # Here we use a tf.keras.layers.Lambda or dummy layer for demonstration.
        # In practice: hub.KerasLayer(bert_path, trainable=True)
        # We simulate a BERT embedding output for demonstration - random tensor output
        self.trainable = True

    def build(self, input_shape):
        # input_shape = [(batch_size, seq_len), (batch_size, seq_len), (batch_size, seq_len)]
        # We simulate bert embeddings by a Dense projection from inputs or random. This is placeholder.
        # A real BERT layer is complex and large; here we use a dummy trainable dense projection.
        # This replacement allows tf.function XLA compilation and demonstration.

        # We create a dense layer simulating BERT output per token:
        self.dense_seq_output = tf.keras.layers.Dense(
            self.output_size, activation=None, name="bert_seq_output_dense"
        )
        super().build(input_shape)

    def call(self, inputs):
        input_ids, input_mask, segment_ids = inputs
        # Simulate an embedding tensor with shape (batch_size, seq_len, output_size)
        # For demonstration, project input_ids embeddings (cast to float32) through dense.
        # Real BERT layer expects tokenized inputs, masks, segments and returns embeddings.
        seq_len = tf.shape(input_ids)[1]

        # Cast inputs to float32 for dummy calculations
        x = tf.cast(input_ids, tf.float32)
        # Expand dims to simulate token embeddings (batch, seq_len, 1)
        x_expanded = tf.expand_dims(x, -1)
        # Broadcast to have last dim = 1, then apply dense to produce output_size dims
        # To simulate token embeddings:
        x_seq = tf.tile(x_expanded, [1, 1, self.output_size])  # shape: (batch, seq_len, output_size)
        # Apply a trainable dense layer to simulate transformation (dummy)
        bert_seq_output = self.dense_seq_output(x_seq)  # (batch, seq_len, output_size)

        if self.pooling == "first":
            # Take first token embedding per batch [CLS]
            pooled = bert_seq_output[:, 0, :]  # (batch, output_size)
        else:  # mean pooling
            # Compute mask as float32, expand dims for broadcasting
            mask = tf.cast(input_mask, tf.float32)
            mask_exp = tf.expand_dims(mask, axis=-1)
            masked_embeddings = bert_seq_output * mask_exp
            sum_embeddings = tf.reduce_sum(masked_embeddings, axis=1)  # sum over seq_len
            lengths = tf.reduce_sum(mask_exp, axis=1) + 1e-10  # avoid div by zero
            pooled = sum_embeddings / lengths  # (batch, output_size)

        return pooled

    def compute_output_shape(self, input_shape):
        # Return shape: (batch_size, output_size)
        return (input_shape[0][0], self.output_size)


class MyModel(tf.keras.Model):
    def __init__(self, max_seq_length=256, num_classes=5, n_fine_tune_layers=3, pooling="first", bert_path=None):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.num_classes = num_classes

        # BERT layer expects inputs: (input_ids, input_mask, segment_ids)
        self.bert_layer = BertLayer(
            n_fine_tune_layers=n_fine_tune_layers,
            pooling=pooling,
            bert_path=bert_path,
            name="bert_layer"
        )

        # Downstream dense layers
        self.dense1 = tf.keras.layers.Dense(256, activation="relu", name="dense_256")
        self.prediction = tf.keras.layers.Dense(num_classes, activation="sigmoid", name="predictions")

    def call(self, inputs):
        # inputs: list or tuple of (input_ids, input_mask, segment_ids)
        pooled_output = self.bert_layer(inputs)  # (batch, 768)
        hidden = self.dense1(pooled_output)      # (batch, 256)
        preds = self.prediction(hidden)          # (batch, num_classes)
        return preds


def my_model_function():
    # Build the MyModel instance with default parameters.
    # bert_path can be None or set to a real model path in practice.
    return MyModel()


def GetInput():
    # Return dummy random input tensors for the model.
    # Inputs: input_ids, input_mask, segment_ids, each shape (B, 256)
    # input_ids are integers [0, vocab_size), masks and segments are 0/1.

    batch_size = 4  # arbitrary small batch size for test
    max_seq_length = 256

    # input_ids: random ints in [0, 30522), standard BERT vocab size; dtype int32
    input_ids = tf.random.uniform(
        shape=(batch_size, max_seq_length), minval=0, maxval=30522, dtype=tf.int32
    )
    # input_mask: binary mask (0 or 1), simulating actual tokens (we take 80% tokens active)
    input_mask = tf.cast(tf.random.uniform(shape=(batch_size, max_seq_length), minval=0, maxval=100, dtype=tf.int32) < 80, tf.int32)
    # segment_ids: binary segment, 0 or 1, random
    segment_ids = tf.cast(tf.random.uniform(shape=(batch_size, max_seq_length), minval=0, maxval=2, dtype=tf.int32), tf.int32)

    return [input_ids, input_mask, segment_ids]


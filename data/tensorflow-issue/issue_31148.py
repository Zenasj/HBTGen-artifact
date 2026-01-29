# tf.random.uniform((B, 32), dtype=tf.int32) ← Input shape inferred from the issue (max_seq_length=32)

import tensorflow as tf

# Placeholder BertLayer to mimic the original BertLayer behavior
# Since the original code depends on TensorFlow Hub BERT module and fine-tune logic,
# here we create a simplified stand-in that returns a 3D tensor resembling BERT output.
class BertLayer(tf.keras.layers.Layer):
    def __init__(self, n_fine_tune_layers=3, pooling="sequence_output", **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        self.n_fine_tune_layers = n_fine_tune_layers
        self.pooling = pooling
        # Output embedding size for BERT base is typically 768
        self.output_dim = 768

    def build(self, input_shape):
        # Normally BERT has many weights, here we use a simple trainable dense to mimic the output
        # We'll simulate sequence output shape: (batch_size, seq_length, 768)
        self.dense = tf.keras.layers.Dense(self.output_dim, activation='linear')
        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        # Inputs are [input_ids, input_mask, segment_ids], each shape (batch_size, seq_length)
        input_ids, input_mask, segment_ids = inputs
        # Shape assumptions:
        batch_size = tf.shape(input_ids)[0]
        seq_length = tf.shape(input_ids)[1]

        # Create a combined token representation by embedding input_ids simply as floats here (for mock)
        # Normally embedding lookup is done, but here we simulate.
        # Simply expand input_ids as float32 and tile to output_dim for demonstration
        token_embeddings = tf.cast(tf.expand_dims(input_ids, -1), tf.float32)  
        token_embeddings = self.dense(token_embeddings)  # (batch_size, seq_length, 768) after linear layer

        # Apply the input_mask to zero out embeddings where mask = 0
        mask_expanded = tf.cast(tf.expand_dims(input_mask, axis=-1), tf.float32)
        token_embeddings *= mask_expanded

        # The 'pooling' parameter as "sequence_output" means we return token-level embeddings
        if self.pooling == "sequence_output":
            return token_embeddings
        elif self.pooling == "pooled_output":
            # Return mean pooled output over tokens (masked)
            sum_embeddings = tf.reduce_sum(token_embeddings, axis=1)
            sum_mask = tf.reduce_sum(mask_expanded, axis=1) + 1e-10
            pooled = sum_embeddings / sum_mask
            return pooled
        else:
            # Default fallback - return sequence output
            return token_embeddings

def build_model(max_seq_length: int = 32):
    # Inputs matching shapes from the issue
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")

    bert_inputs = [in_id, in_mask, in_segment]

    bert_output = BertLayer(n_fine_tune_layers=3, pooling="sequence_output")(bert_inputs)
    # This has shape (batch_size, seq_length, 768)

    dense = tf.keras.layers.Dense(256, activation='relu')(bert_output)
    pred = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
    # Output shape: (batch_size, seq_length, 1)

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

class MyModel(tf.keras.Model):
    def __init__(self, max_seq_length=32, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.max_seq_length = max_seq_length
        # Instantiate the BERT-based model
        self.inner_model = build_model(max_seq_length)

    def call(self, inputs):
        # inputs should be a list or tuple of [input_ids, input_mask, segment_ids]
        return self.inner_model(inputs)

def my_model_function():
    return MyModel()

def GetInput():
    max_seq_length = 32
    batch_size = 4  # Provide a small batch size

    # Generate random integer inputs in the range [0, 10000) as typical token IDs
    input_ids = tf.random.uniform([batch_size, max_seq_length], minval=0, maxval=10000, dtype=tf.int32)
    input_masks = tf.ones([batch_size, max_seq_length], dtype=tf.int32)  # All tokens valid (no mask)
    segment_ids = tf.zeros([batch_size, max_seq_length], dtype=tf.int32)  # Single segment

    return [input_ids, input_masks, segment_ids]


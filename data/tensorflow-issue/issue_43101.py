# tf.random.uniform((B, None), dtype=tf.int32) ← With B=32 as batch size, variable length sequences of integers (tokens)

import tensorflow as tf

class MyModel(tf.keras.Model):
    """Model combining Embedding, LSTM with masking, and Dense output, plus custom train_step with gradient stacking."""

    def __init__(self):
        super().__init__()
        # Embedding with mask_zero=True to produce masking for padded tokens (0s)
        self.embedding = tf.keras.layers.Embedding(input_dim=200, output_dim=128, mask_zero=True)
        # LSTM layer that supports masking implicitly
        self.lstm = tf.keras.layers.LSTM(128)
        # Final classification dense layer with softmax for 2 classes
        self.classifier = tf.keras.layers.Dense(2, activation='softmax')

        # Use typical metrics for tracking during fit
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.accuracy = tf.keras.metrics.CategoricalAccuracy()
    
    def call(self, inputs, training=False):
        # inputs shape: (batch_size, seq_len)
        x = self.embedding(inputs)  # Shape: (batch_size, seq_len, 128), mask propagated automatically
        x = self.lstm(x)            # Shape: (batch_size, 128)
        out = self.classifier(x)    # Shape: (batch_size, 2)
        return out

    def train_step(self, data):
        """Custom training step that stacks gradients over 4 sub-batches using a tf.while_loop over tensors.

        This mimics the gradient stacking gradient accumulation approach described in the issue.

        Expected data: tuple (inputs, labels), inputs shape (batch, seq_len).
        """
        inputs, y_true = data
        batch_size = tf.shape(inputs)[0]
        quarter = batch_size // 4

        # Get gradients on the first quarter
        gradients = self._get_gradients(inputs[:quarter], y_true[:quarter])

        # Function to process each quarter and accumulate gradients
        def process_quarter(idx, grads_accum):
            start = idx * quarter
            end = (idx + 1) * quarter
            grads_local = self._get_gradients(inputs[start:end], y_true[start:end])
            # Add gradients elementwise
            grads_accum = [self._add_gradients(a, b) for a, b in zip(grads_accum, grads_local)]
            return idx + 1, grads_accum

        # Iterate over 2nd to 4th quarter (indices 1 to 3 inclusive)
        _, gradients = tf.while_loop(
            cond=lambda idx, _: idx < 4,
            body=process_quarter,
            loop_vars=[tf.constant(1), gradients],
            parallel_iterations=1
        )

        # Apply accumulated gradients
        grads_and_vars = zip(gradients, self.trainable_variables)
        self.optimizer.apply_gradients(grads_and_vars)

        # Update and return metrics
        y_pred = self(inputs, training=True)
        self.compiled_loss(y_true, y_pred)
        self.accuracy.update_state(y_true, y_pred)
        return {'loss': self.compiled_loss.result(), 'categorical_accuracy': self.accuracy.result()}

    def _get_gradients(self, inputs, y_true):
        """Utility to compute gradients on (inputs, y_true) batch."""
        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)
            loss = self.compiled_loss(y_true, y_pred)
        return tape.gradient(loss, self.trainable_variables)

    @staticmethod
    def _add_gradients(grad_a, grad_b):
        """Add two gradient objects, which can be Tensors or IndexedSlices."""
        if not isinstance(grad_b, type(grad_a)):
            raise TypeError('Adding gradients with different types.')
        if isinstance(grad_a, tf.Tensor):
            return grad_a + grad_b
        if isinstance(grad_a, tf.IndexedSlices):
            values = tf.concat([grad_a.values, grad_b.values], axis=0)
            indices = tf.concat([grad_a.indices, grad_b.indices], axis=0)
            return tf.IndexedSlices(values, indices, grad_a.dense_shape)
        # Fallback, just add directly if possible
        return grad_a + grad_b

def my_model_function():
    """Factory to build the custom MyModel, ready for training with gradient stacking."""
    model = MyModel()
    # Compile the model with suitable optimizer and loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )
    return model

def GetInput():
    """Return a batch of padded token sequences (batch_size=32) with random token ids in [1, 199], padding with zeros.

    Shape: (32, variable_length)
    dtype: tf.int32
    """
    import numpy as np

    batch_size = 32
    # Generate sequences of length from 8 to 32 tokens
    seq_lengths = np.random.randint(8, 33, size=batch_size)
    max_len = seq_lengths.max()

    # Prepare numpy array filled with zeros (padding token)
    input_batch = np.zeros((batch_size, max_len), dtype=np.int32)

    for i, length in enumerate(seq_lengths):
        # Random tokens from 1 to 199 (0 is reserved for padding)
        input_batch[i, :length] = np.random.randint(1, 200, size=length)

    return tf.convert_to_tensor(input_batch, dtype=tf.int32)


# tf.random.uniform((N_samples, N_dim), dtype=tf.float32) ← Input shape is (batch size, feature dim)
import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        """
        Constructs a model replicating the DenseCov + final Dense layer architecture.
        The DenseCov layer tries to minimize the trace of covariance matrices over segments
        of the weight matrix, with segments specified. The loss is added via add_loss.
        
        Segment-related logic is preserved though in the recreations from the issue 
        segments are all zero, effectively one segment, but the structure kept.
        """
        super().__init__()

        # Parameters from issue adjusted for feasible smaller size
        self.N_class = 500         # number of classes / units in DenseCov
        self.N_domains = 1         # single domain segment for simplicity
        self.N_dim = 10            # feature dimension in input
        self.segments = np.random.randint(0, self.N_domains, self.N_class)
        self.segments = np.sort(self.segments)

        # Store segment ids as constant for RaggedTensor rowids construction
        self.segment_ids = tf.constant(self.segments, dtype=tf.int64, name='segment_ids')

        # DenseCov layer components
        # We replicate DenseCov behavior inside this model

        # DenseCov layer: units=N_class, no bias, kernel initialized to ones
        self.dense_cov_kernel = self.add_weight(
            name="dense_cov_kernel",
            shape=(self.N_dim, self.N_class),
            initializer=tf.keras.initializers.Ones(),
            trainable=True,
            dtype=tf.float32
        )

        # Final dense layer following DenseCov - maps N_class to scalar output
        self.final_dense = tf.keras.layers.Dense(1, name='final_dense')

        # On build an additive noise adjustment is applied to dense_cov_kernel to distort matrix
        self._built = False

    def build(self, input_shape):
        super().build(input_shape)
        # Perturb kernel weights slightly with random normal noise (to simulate distortion)
        noise = np.random.randn(self.N_dim, self.N_class).astype(np.float32)
        self.dense_cov_kernel.assign_add(noise)
        self._built = True

    def call(self, inputs, training=None):
        # Forward pass through DenseCov layer logic
        # 1) Compute logits: inputs @ dense_cov_kernel
        # 2) Compute loss related to trace of covariance matrices of weight segments
        #    using tf.RaggedTensor.from_value_rowids and tf.map_fn

        logits = tf.matmul(inputs, self.dense_cov_kernel)  # Shape (batch, N_class)

        # RaggedTensor construction:
        # RaggedTensor segments weights (dense_cov_kernel.T) by segment_ids (rowids)
        # kernel shape: (N_dim, N_class)
        # We want to segment by segments per column, so transpose to (N_class, N_dim)
        kernel_transposed = tf.transpose(self.dense_cov_kernel)  # shape (N_class, N_dim)

        # Use RaggedTensor.from_value_rowids: values=kernel_transposed, rowids=segment_ids
        # This groups rows of kernel_transposed into ragged segments
        W_ragged = tf.RaggedTensor.from_value_rowids(values=kernel_transposed, value_rowids=self.segment_ids)

        # Compute means over each segment along axis=1 (within each ragged segment rows)
        means = tf.reduce_mean(W_ragged, axis=1)  # ragged: shape (num_segments, N_dim)

        # Center each segment by subtracting means (broadcast)
        W_centred = W_ragged - tf.expand_dims(means, axis=1)  # ragged subtraction

        # For each segment matrix (segment rows x N_dim), compute covariance matrix:
        # cov = X^T @ X / (n - 1)
        def segment_covariance(x):
            # x shape: (segment_length, N_dim)
            n = tf.cast(tf.shape(x)[0], tf.float32)
            cov = tf.matmul(x, x, transpose_a=True) / (n - 1.0)
            return cov

        cov_matrix = tf.map_fn(segment_covariance, W_centred, fn_output_signature=tf.float32)  # (num_segments, N_dim, N_dim)

        # cov_matrix is dense tensor since tf.map_fn returns stack

        # Compute trace of each covariance matrix:
        traces = tf.linalg.trace(cov_matrix)  # shape (num_segments,)

        # Loss is mean of traces across segments
        loss = tf.reduce_mean(traces, name='cov_trace_loss')

        # Add loss for optimization
        self.add_loss(loss)

        # Pass logits through final dense layer for downstream task predictions
        output = self.final_dense(logits)  # shape (batch, 1)

        return output

def my_model_function():
    # Return instance of MyModel with necessary initializations
    model = MyModel()
    # It will be built automatically on call with input shape
    return model

def GetInput():
    # Return a batch of random inputs matching input shape (N_samples=10 for demonstration, N_dim=10)
    # Use dtype float32 as per typical TF float default
    N_samples = 10
    N_dim = 10
    return tf.random.uniform((N_samples, N_dim), dtype=tf.float32)


# tf.random.uniform((15, 8, 150), dtype=tf.float32) ← inferred input shape from placeholders and cudnn_gru calls

import tensorflow as tf

# Assumptions:
# - The original issue and examples revolve around using tf.contrib.cudnn_rnn.CudnnGRU layers,
#   sometimes in control flow (tf.while_loop or tf.cond), which cause internal errors.
# - There were two similar cudnn GRU models demonstrated, one with num_units=150, the other num_units=149,
#   used conditionally.
# - We fuse these two models into a single MyModel class, holding both submodules,
#   running both on the input tensor, then comparing their outputs by shape and content.
# - Since the cudnn GRU API may differ in TF2, we keep it in TF1 style using tf.contrib where possible,
#   assuming compatibility with TF 2.20.0 XLA compilation for example.
# - Input shape: [seq_len=15, batch_size=8, feature_dim=150], consistent with placeholders.

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.num_units1 = 150
        self.num_units2 = 149
        
        # The cudnn GRU instances, one with 150 units, one with 149 units
        # Using tf.contrib.cudnn_rnn.CudnnGRU as in the issue, with one layer each.
        self.gru1 = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1, num_units=self.num_units1, input_size=self.num_units1)
        self.gru2 = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1, num_units=self.num_units2, input_size=self.num_units1)

        # Variables for initial states for both models, per example code, shape [1, batch_size, num_units]
        # Use tf.Variable instead of tf.get_variable for better TF2 compatibility.
        # 1 = num_layers, 8 = batch_size (fixed), num_units as per model
        batch_size = 8
        self.init_fw_1 = tf.Variable(tf.zeros([1, batch_size, self.num_units1]), trainable=False, name='init_fw_1')
        self.init_fw_2 = tf.Variable(tf.zeros([1, batch_size, self.num_units2]), trainable=False, name='init_fw_2')
        
    def call(self, inputs):
        '''
        inputs: tf.Tensor of shape [seq_len, batch_size, feature_dim=150], dtype=tf.float32
        Returns:
            A dictionary with outputs and comparison result (bool tensor).
        '''
        # Run both cudnn GRU models on the same inputs
        # According to TF1 cudnn GRU API:
        # output, output_h = gru(inputs, initial_state)
        out1, _ = self.gru1(inputs, initial_state=self.init_fw_1)
        out2, _ = self.gru2(inputs, initial_state=self.init_fw_2)
        
        # Since units differ (150 vs 149), shapes differ along last dim:
        # out1 shape: [seq_len, batch_size, 150]
        # out2 shape: [seq_len, batch_size, 149]
        # For comparison, truncate or pad smaller dim to equalize or just compare shape mismatch.
        # Here, for demonstration, compare shapes (expected different), then compare truncated slices.

        min_units = min(out1.shape[-1], out2.shape[-1])
        
        out1_trunc = out1[:, :, :min_units]
        out2_trunc = out2[:, :, :min_units]

        # Compute elementwise difference
        diff = tf.abs(out1_trunc - out2_trunc)

        # Define a tolerance to consider outputs nearly equal
        tol = 1e-5
        outputs_equal = tf.reduce_all(diff < tol)

        # Return a dictionary (or tuple) of results
        # For simplicity, output dictionary with original outputs and bool comparison
        return {
            'output1': out1,
            'output2': out2,
            'outputs_equal': outputs_equal,
            'outputs_diff': diff
        }

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching expected shape of MyModel
    # Shape: [seq_len=15, batch_size=8, feature_dim=150], float32
    # Use uniform distribution as per original snippets
    return tf.random.uniform(shape=[15, 8, 150], dtype=tf.float32)


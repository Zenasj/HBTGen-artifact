# tf.random.uniform((B, T, 512), dtype=tf.float32) ← Input shape: (batch_size, time_steps, 512)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # CuDNNGRU layer with 384 units, returns sequences and state
        self.cudnn_gru = tf.keras.layers.GRU(
            384,
            return_sequences=True,
            return_state=True,
            reset_after=True,
            recurrent_activation='sigmoid',
            # In TF 2.x, CuDNNGRU is deprecated and merged into GRU with 
            # reset_after=True and recurrent_activation='sigmoid'
            # This ensures it uses the CuDNN kernels on GPU if available
        )
        # For the first model from chunk 1/2:
        # Split GRU outputs into two halves and use separate Dense stacks
        self.dense_256_relu_1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense_128_1 = tf.keras.layers.Dense(128)
        self.dense_256_relu_2 = tf.keras.layers.Dense(256, activation='relu')
        self.dense_128_2 = tf.keras.layers.Dense(128)

        # For the simplified model from chunk 2/3 (single logits path)
        self.dense_256_relu_single = tf.keras.layers.Dense(256, activation='relu')
        self.dense_128_single = tf.keras.layers.Dense(128)

    def call(self, inputs, training=False):
        """
        inputs: shape (batch_size, time_steps, 512)
        Return: a dictionary of outputs for both variants and a 'comparison' boolean tensor 
        that tests if outputs are "close".
        
        This fuses the two described models:
        - original split GRU outputs + two dense paths concatenated (outputs_part1)
        - simplified single logits output (output_single)
        
        Then compares the outputs for debugging / analysis as in issue context.
        """
        rnn_outputs, rnn_state = self.cudnn_gru(inputs, training=training)

        # Original model split output path
        h_1, h_2 = tf.split(rnn_outputs, num_or_size_splits=2, axis=-1)
        logits_1 = self.dense_128_1(self.dense_256_relu_1(h_1))
        logits_2 = self.dense_128_2(self.dense_256_relu_2(h_2))
        outputs_part1 = tf.concat([logits_1, logits_2], axis=-1)  # shape (B, T, 256)

        # Simplified single logits output (from chunk 2/3)
        outputs_single = self.dense_128_single(self.dense_256_relu_single(rnn_outputs))  # shape (B, T, 128)

        # For the sake of comparison, reduce outputs_part1 to 128 channels by average pooling over last dim pairs
        # This is an approximation to allow shape matching for comparison (256 -> 128)
        outputs_part1_reduced = 0.5 * (outputs_part1[..., :128] + outputs_part1[..., 128:])

        # Compare shapes: outputs_part1_reduced and outputs_single both (B,T,128)
        # Use element-wise close comparison with tolerance for float differences
        comparison = tf.reduce_all(
            tf.abs(outputs_part1_reduced - outputs_single) < 1e-4
        )

        # Return a dictionary of these outputs and the comparison boolean
        # This fused output reflects the original issue discussion about both models
        return {
            'outputs_part1': outputs_part1,
            'outputs_single': outputs_single,
            'outputs_part1_reduced': outputs_part1_reduced,
            'comparison_result': comparison
        }


def my_model_function():
    # Return an instance of MyModel, no pretrained weights to load
    return MyModel()

def GetInput():
    # Return a random tensor matching the input expected by MyModel
    # batch_size and time_steps from issue examples for typical benchmarking
    batch_size = 256
    time_steps = 750
    # Input shape: (batch_size, time_steps, 512)
    return tf.random.uniform((batch_size, time_steps, 512), dtype=tf.float32)


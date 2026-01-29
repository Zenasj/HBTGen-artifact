# tf.random.uniform((B, 10, 80), dtype=tf.float32) ← Inferred input shape based on dummy data: (batch_size, 10, 80)

import tensorflow as tf
import numpy as np

loss_tracker = tf.keras.metrics.Mean(name="loss")

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Dense layer projecting from input_dim=200 to 80 units
        self.Dense = tf.keras.layers.Dense(80)
        # Mean Squared Error loss with no reduction (per example)
        self.MSE = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    def train_step(self, data):
        # data shape: [batch_size, 10, 80]

        # Get batch size dynamically from input tensor shape
        batch_size = tf.shape(data)[0]
        max_length = 10

        # Dummy input tensor for loop: shape [batch_size, 1, 200]
        dummy_inputs = tf.zeros([batch_size, 1, 200], dtype=tf.float32)

        # Initial output tensor: empty along time dimension, shape [batch_size, 0, 80]
        gen_outputs = tf.zeros([batch_size, 0, 80], dtype=tf.float32)

        # Loop variable i and loop body function
        i_start = tf.constant(0, dtype=tf.int32)

        def body(i, input, output_full):
            # Apply Dense to input shape [batch_size, 1, 200] → output shape [batch_size, 1, 80]
            output_single = self.Dense(input)
            # Concatenate output to accumulated outputs along time axis (axis=1)
            output_full = tf.concat([output_full, output_single], axis=1)
            i_next = i + 1
            return i_next, input, output_full

        with tf.GradientTape() as tape:
            # Run while loop from i=0 to max_length (10)
            _, _, gen_data = tf.while_loop(
                cond=lambda i, input, output_full: tf.less(i, max_length),
                body=body,
                loop_vars=(i_start, dummy_inputs, gen_outputs),
                shape_invariants=(
                    i_start.get_shape(),
                    tf.TensorShape([None, 1, 200]),
                    tf.TensorShape([None, None, 80]),
                ),
            )
            # Compute loss per example (batch element), shape [batch_size, 10, 80]
            # MSE with reduction NONE returns squared error per element; take mean over last dims
            loss_per_elem = self.MSE(data, gen_data)  # shape: [batch_size, 10]
            # Reduce over time steps and features axis to get scalar losses per batch element
            loss = tf.reduce_mean(loss_per_elem)

        # Compute gradients and update trainable variables
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update the loss tracker (mean over batches)
        loss_tracker.update_state(loss)
        return {"loss": loss_tracker.result()}


def my_model_function():
    # Create instance of MyModel and compile with Adam optimizer
    model = MyModel()
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer)
    return model


def GetInput():
    # Return a random tensor input matching expected input shape of data:
    # shape: (batch_size, sequence_length, features) = (32, 10, 80)
    batch_size = 32
    seq_len = 10
    features = 80
    # Use uniform random data as input examples
    return tf.random.uniform((batch_size, seq_len, features), dtype=tf.float32)


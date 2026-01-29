# tf.random.uniform((B, 4), dtype=tf.float32) ← The input shape is (batch_size, 4) consistent with the example code

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, output_dim: int = 2, decay_rate: float = 0.99):
        super(MyModel, self).__init__()
        self.output_dim = output_dim
        self.decay = decay_rate
        # Encoder 1 and Encoder 2 have identical architectures: Dense layer with ReLU activation on input shape (4,)
        self.encoder_1 = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(4,), dtype=tf.float32),
            tf.keras.layers.Dense(units=self.output_dim, activation='relu')
        ])
        self.encoder_2 = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(4,), dtype=tf.float32),
            tf.keras.layers.Dense(units=self.output_dim, activation='relu')
        ])

    def call(self, x: tf.Tensor, training: bool = False):
        # Forward pass returns encoder_1 output
        return self.encoder_1(x, training=training)

    def train_step(self, data):
        # Custom training step that updates encoder_1 using loss to encoder_2 outputs, then updates encoder_2 weights as EMA of encoder_1 weights
        x = data
        # Compute encoder_2 output as target (no gradient update on encoder_2 here)
        out_2 = self.encoder_2(x, training=True)

        with tf.GradientTape() as tape:
            out_1 = self.encoder_1(x, training=True)
            # Use compiled_loss to compute loss (e.g., mse between encoder_1 and encoder_2 outputs)
            loss = self.compiled_loss(out_1, out_2)

        # Compute gradients for encoder_1 trainable weights
        enc_1_grads = tape.gradient(loss, self.encoder_1.trainable_weights)

        # Apply gradients to encoder_1 weights
        self.optimizer.apply_gradients(zip(enc_1_grads, self.encoder_1.trainable_weights))

        # --- Workaround for the TF issue with set_weights inside train_step ---
        # Instead of calling set_weights directly with tensors (which triggers TypeError),
        # convert updated weights to numpy arrays explicitly before calling set_weights.
        enc_1_weights = self.encoder_1.get_weights()
        enc_2_weights = self.encoder_2.get_weights()

        # Compute exponentially moving average of weights
        new_weights = []
        for w1, w2 in zip(enc_1_weights, enc_2_weights):
            # w1, w2 are NumPy arrays from get_weights
            updated_w = self.decay * w2 + (1 - self.decay) * w1
            new_weights.append(updated_w)

        # Update encoder_2 weights safely with NumPy arrays
        self.encoder_2.set_weights(new_weights)

        # Update metrics (if any)
        self.compiled_metrics.update_state(out_1, out_2)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


def my_model_function():
    # Instantiate the model with default params (output_dim=2, decay=0.99)
    return MyModel()

def GetInput():
    # Return random input tensor matching expected input shape (batch_size, 4)
    # Let's assume batch size of 10 for testing
    return tf.random.uniform((10, 4), dtype=tf.float32)


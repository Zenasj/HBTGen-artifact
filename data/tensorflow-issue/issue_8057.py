# tf.random.uniform((B, H, W, C), dtype=...) ← Input shape is ambiguous in the issue, so we will assume a generic 4D input shape.
# For demonstration, assume input is batch of sequences with shape (batch_size, seq_length, feature_dim).
# We'll pick (32, 100, 64) as a reasonable default example.

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # For this example, assume input shape is (batch_size, seq_length, feature_dim)
        # We build a simple model with an LSTM and a Dense output layer as a placeholder.
        self.lstm = tf.keras.layers.LSTM(128, return_sequences=True)
        self.dense = tf.keras.layers.Dense(10)  # For example, 10-class output logits

        # We also create an Adam optimizer internally, exposing beta1_power and beta2_power 
        # variables manually, as described in the issue.

        # Use this optimizer, mimicking TF1 behavior where beta*_power are separate variables.
        self.optimizer = tf.keras.optimizers.Adam()

        # In TF2 Keras optimizer, beta1_power and beta2_power are tracked internally as variables:
        # To emulate the issue environment, we create and track them manually.
        self._beta1_power = tf.Variable(self.optimizer.beta_1, trainable=False, name='beta1_power')
        self._beta2_power = tf.Variable(self.optimizer.beta_2, trainable=False, name='beta2_power')

    def call(self, inputs, training=False):
        x = self.lstm(inputs)
        logits = self.dense(x)

        # For demonstration, output difference between manually tracked beta powers and
        # optimizer internal values as representation of "comparison" (following the merged model requirement).
        # Note: TF2 Adam optimizer variables are not publicly exposed the same way,
        # so this is a heuristic to show the concept.
        # In practice, these variables differ only by updates during training.
        optimizer_beta1_power = self.optimizer.get_slot(self.optimizer.variables()[0], 'm') if self.optimizer.variables() else None
        # The following outputs a numeric tensor showing difference from tracked power variables (may be None).
        beta1_diff = tf.constant(0.0)
        beta2_diff = tf.constant(0.0)
        if optimizer_beta1_power is not None:
            try:
                beta1_diff = tf.abs(self._beta1_power - self.optimizer.beta_1)
                beta2_diff = tf.abs(self._beta2_power - self.optimizer.beta_2)
            except Exception:
                pass  # fallback if access not possible

        return logits, beta1_diff, beta2_diff

def my_model_function():
    # Return an instance of MyModel.
    # No additional weights initialization needed beyond constructor.
    return MyModel()

def GetInput():
    # Return a random tensor input matching (batch_size, seq_length, feature_dim)
    # matching what MyModel expects.

    batch_size = 32
    seq_length = 100
    feature_dim = 64

    # Use float32 uniformly distributed tensor as dummy input.
    return tf.random.uniform((batch_size, seq_length, feature_dim), dtype=tf.float32)


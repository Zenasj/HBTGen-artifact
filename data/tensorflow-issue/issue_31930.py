# tf.random.uniform((B, 1), dtype=tf.float32) ← Based on example x_train shape (1,), single feature input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Dense layer without activation (outputs logits)
        self.dense_logits = tf.keras.layers.Dense(
            1, use_bias=False, trainable=False,
            kernel_initializer=tf.constant_initializer(1.0)
        )
        # Dense layer with sigmoid activation inside
        self.dense_sigmoid = tf.keras.layers.Dense(
            1, use_bias=False, trainable=False,
            activation='sigmoid',
            kernel_initializer=tf.constant_initializer(1.0)
        )
        # Dense layer without activation + explicit sigmoid later
        self.dense_no_activation = tf.keras.layers.Dense(
            1, use_bias=False, trainable=False,
            kernel_initializer=tf.constant_initializer(1.0)
        )

    def call(self, inputs):
        """
        Forward pass returns a dictionary comparing the three output variants:
          - logits output (no activation)
          - separate sigmoid output
          - sigmoid inside Dense output

        Also returns results of BinaryCrossentropy loss with appropriate from_logits flags on these outputs,
        to illustrate the differences described in the issue.

        The output shape: A dictionary of tensors with shape (batch_size, 1)
        """

        # Compute logits (no activation)
        logits = self.dense_logits(inputs)  # shape (B,1)

        # Compute sigmoid explicitly from logits
        sigmoid_explicit = tf.sigmoid(self.dense_no_activation(inputs))  # (B,1)

        # Compute sigmoid inside Dense activation
        sigmoid_dense = self.dense_sigmoid(inputs)  # (B,1)

        # Create loss functions with different from_logits flags
        # This simulates the different behaviors described in the issue:
        bce_from_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        bce_no_logits = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

        # For demonstration, assume labels=0 (like in the example) for batch
        labels = tf.zeros_like(logits)

        # Calculate losses according to different scenarios
        loss_logits = bce_from_logits(labels, logits)       # expects logits -> no sigmoid applied internally
        loss_sigmoid_explicit = bce_no_logits(labels, sigmoid_explicit)  # expects probability inputs
        loss_sigmoid_dense = bce_no_logits(labels, sigmoid_dense)        # expects probability inputs

        # Return dictionary to observe outputs and losses side-by-side
        return {
            'logits': logits,
            'sigmoid_explicit': sigmoid_explicit,
            'sigmoid_dense': sigmoid_dense,
            'loss_logits': tf.expand_dims(loss_logits, axis=-1),
            'loss_sigmoid_explicit': tf.expand_dims(loss_sigmoid_explicit, axis=-1),
            'loss_sigmoid_dense': tf.expand_dims(loss_sigmoid_dense, axis=-1),
        }

def my_model_function():
    """
    Returns an instance of MyModel.
    The Dense layers use constant initializer to match the fixed weight [1] in original example,
    and weights are set as non-trainable for comparison consistency.
    """
    model = MyModel()
    # The weights are initialized with constant 1.0 due to kernel_initializer above,
    # reflecting the weight [1] used in the original code example.
    return model

def GetInput():
    """
    Returns a sample input tensor compatible with MyModel.
    Matches the input shape from the example: shape=(batch_size=1, features=1).
    Random uniform float32 values between 500 and 1500 to demonstrate logits with large magnitude.
    """
    return tf.random.uniform(shape=(1, 1), minval=500.0, maxval=1500.0, dtype=tf.float32)


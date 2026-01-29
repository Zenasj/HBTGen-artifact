# tf.random.uniform((B=10, 100), dtype=tf.float32) and (B=10, 10)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define generator and discriminator as in the issue example
        self.gen = tf.keras.layers.Dense(100)
        self.dis = tf.keras.layers.Dense(1)

        # We create submodels analogous to dis_model and combined_model
        # Input shapes based on example: noise (batch, 10), x (batch, 100)
        noise_in = tf.keras.Input(shape=(10,))
        x_in = tf.keras.Input(shape=(100,))

        # Discriminator model: x -> dis
        dis_out = self.dis(x_in)
        self.dis_model = tf.keras.Model(x_in, dis_out, name="dis_model")
        self.dis_model.compile(optimizer='rmsprop', loss='mse')

        # Combined model: noise -> gen -> dis (dis_model)
        # Important: set dis_model.trainable = False before creating combined_model so dis weights freeze
        self.dis_model.trainable = False
        combined_out = self.dis_model(self.gen(noise_in))
        self.combined_model = tf.keras.Model(noise_in, combined_out, name="combined_model")
        self.combined_model.compile(optimizer='rmsprop', loss='mse')

        # After compiling combined_model, set dis_model.trainable=True to suppress warnings as suggested
        self.dis_model.trainable = True

    def call(self, inputs, training=False):
        """
        Forward pass takes a tuple of inputs (x_input, noise_input)
        and returns a dictionary with dis_model output, combined_model output,
        and also a boolean tensor indicating if dis_model.trainable weights
        and collected_trainable_weights states are consistent.
        """
        x_input, noise_input = inputs

        # Outputs of submodels
        dis_out = self.dis_model(x_input, training=training)
        combined_out = self.combined_model(noise_input, training=training)

        # For comparison example, we compare trainable weights sets for demonstration:
        # Note: In the issue, the warning relates to mismatch between
        # _collected_trainable_weights and trainable_weights,
        # but these are internal protected members, so just illustrate pts.

        # Check if dis_model.trainable_weights is consistent with _collected_trainable_weights
        # This is a proxy for the warning condition in the original issue.
        collected = getattr(self.dis_model, "_collected_trainable_weights", None)
        trainable = self.dis_model.trainable_weights

        # If collected_weights is None (may happen if not compiled), set flag False
        if collected is None:
            consistency = tf.constant(False)
        else:
            # We check if the sets of variables are equal by object identity
            set_collected = set(collected)
            set_trainable = set(trainable)
            consistency = tf.reduce_all(
                tf.constant([v in set_collected for v in set_trainable])) and \
                          tf.reduce_all(
                tf.constant([v in set_trainable for v in set_collected])
            )
            # Convert python bool to tensor scalar
            consistency = tf.constant(consistency)

        return {
            "dis_output": dis_out,
            "combined_output": combined_out,
            "trainable_consistency": consistency
        }

def my_model_function():
    # Return an instance of MyModel with generator and discriminator initialized
    # and compiled exactly per the issue example for reproducibility.
    return MyModel()

def GetInput():
    # Returns a tuple of inputs corresponding to (x_input, noise_input) expected by MyModel.call
    x_input = tf.random.uniform((10, 100), dtype=tf.float32)
    noise_input = tf.random.uniform((10, 10), dtype=tf.float32)
    return (x_input, noise_input)


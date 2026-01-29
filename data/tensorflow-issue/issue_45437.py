# tf.random.uniform((B, 100), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define m1: a sequential model with Input, Dense(20), Dense(10)
        # Using Functional API for clarity to explicitly include Input layer
        inputs_m1 = tf.keras.Input(shape=(100,), name="m1_input")
        x = tf.keras.layers.Dense(20, name="dense_20")(inputs_m1)
        out_m1 = tf.keras.layers.Dense(10, name="dense_10")(x)
        self.m1 = tf.keras.Model(inputs=inputs_m1, outputs=out_m1, name="m1")

        # Define m2: sequential with Input, m1 as a layer, Dense(5)
        # Use Functional API because embedding a model as a layer is simplest this way
        inputs_m2 = tf.keras.Input(shape=(100,), name="m2_input")
        out_m1_in_m2 = self.m1(inputs_m2)
        out_m2 = tf.keras.layers.Dense(5, name="dense_5")(out_m1_in_m2)
        self.m2 = tf.keras.Model(inputs=inputs_m2, outputs=out_m2, name="m2")

    def call(self, inputs, training=False):
        """
        For this combined model, the forward pass returns:
        - The full output of m2 (i.e. output after Dense(5))
        - The intermediate Dense(10) output from m1 when embedded into m2
        - Also returns a boolean indicating if the feature extraction from m1 matches 
          what would be obtained by directly calling m1 on the input (sanity check comparison).
        
        Because the issue centered around feature extraction from nested models,
        this combined call provides a way to get both outputs and compare internally.
        """
        # Full output from m2
        out_m2 = self.m2(inputs, training=training)

        # Output of the Dense(10) layer inside m1 embedded in m2 (via m1 model call)
        feature_from_m1_in_m2 = self.m1(inputs, training=training)

        # Also get feature from m1 directly (same inputs)
        feature_m1_direct = self.m1(inputs, training=training)

        # Compare if feature_from_m1_in_m2 matches feature_m1_direct
        # This is to emulate the "sanity check" or inspection in the issue discussion
        match = tf.reduce_all(tf.abs(feature_from_m1_in_m2 - feature_m1_direct) < 1e-6)

        return {
            "m2_output": out_m2,
            "m1_feature_in_m2": feature_from_m1_in_m2,
            "feature_match": match
        }


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random input tensor shaped (batch_size=1, 100), dtype float32
    # Matches the Input shape used in m1 and m2
    return tf.random.uniform(shape=(1, 100), dtype=tf.float32)


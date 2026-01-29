# tf.random.uniform((6, 1), dtype=tf.float32) for float inputs and tf.constant with shape (6, 1), dtype=tf.string for string inputs
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Encoder A inputs: two float features
        self.encoder_a_input_shapes = [(1,), (1,)]
        self.encoder_a_dense = tf.keras.layers.Dense(4, activation="relu")
        
        # Encoder B inputs: one string categorical feature + one float
        self.string_lookup = tf.keras.layers.StringLookup(
            vocabulary=("a", "b", "c"), output_mode="one_hot"
        )
        self.encoder_b_dense = tf.keras.layers.Dense(4, activation="relu")

        # Final output layer 
        self.final_dense = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        """
        Inputs is expected to be a tuple/list of two dicts:
          inputs[0]: dict for encoder A inputs (keys: "a0", "a1"), float32 tensors shape (batch,1)
          inputs[1]: dict for encoder B inputs (keys: "b1", and some string input), 
                     with the string input name not fixed but expected to be string tensor shape (batch, 1)
        """
        inputs_a, inputs_b = inputs

        # Encoder A pathway
        a0 = inputs_a["a0"]
        a1 = inputs_a["a1"]
        concat_a = tf.concat([a0, a1], axis=-1)
        encoded_a = self.encoder_a_dense(concat_a)

        # Encoder B pathway
        # We need to detect the string input key (the one with dtype string)
        string_key = None
        for k, v in inputs_b.items():
            if v.dtype == tf.string:
                string_key = k
                break
        if string_key is None:
            raise ValueError("No string input found in encoder B inputs")

        bx = inputs_b[string_key]
        b1 = inputs_b["b1"]
        encoded_bx = self.string_lookup(bx)
        concat_b = tf.concat([encoded_bx, b1], axis=-1)
        encoded_b = self.encoder_b_dense(concat_b)

        # Compute absolute difference and final probability prediction
        diff = tf.abs(encoded_a - encoded_b)
        prob = self.final_dense(diff)
        return prob

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Create example inputs that match the expected input structure and dtypes

    batch_size = 6

    # Encoder A inputs (float32 tensors with shape (batch,1))
    x_a = {
        "a0": tf.random.uniform((batch_size, 1), dtype=tf.float32),
        "a1": tf.random.uniform((batch_size, 1), dtype=tf.float32)
    }

    # Encoder B inputs
    # One float32 input with shape (batch,1)
    b1 = tf.random.uniform((batch_size, 1), dtype=tf.float32)

    # One string input with shape (batch,1) chosen from vocabulary ("a", "b", "c")
    # Use e.g. a fixed pattern for testing
    bx_values = tf.constant(["a", "b", "c", "a", "b", "a"])
    bx = tf.reshape(bx_values, (batch_size, 1))

    x_b = {
        "b1": b1,
        "bx": bx,  # This simulates the non-fixed string input name
    }

    return [x_a, x_b]


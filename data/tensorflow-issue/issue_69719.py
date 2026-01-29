# tf.random.uniform((B, 10), dtype=tf.float32) ← input shape inferred from data X = np.random.rand(100, 10)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, scaling_vector):
        """
        A fused MLP + Rescaling model that encapsulates the behavior described:
        - Three Dense layers with relu except last
        - A Rescaling layer after the last Dense layer using a fixed scaling vector

        The Rescaling layer applies elementwise multiplication by the given scaling vector,
        fixing issues that arise from saving/loading the standard Rescaling layer with vector inputs.

        Args:
          scaling_vector: a 1D tensor or list/array with length matching output dim of last Dense (5)
        """
        super().__init__()
        # Define MLP layers
        self.dense1 = tf.keras.layers.Dense(100, activation="relu", name="layer1")
        self.dense2 = tf.keras.layers.Dense(10, activation="relu", name="layer2")
        self.dense3 = tf.keras.layers.Dense(5, name="layer3")
        
        # Instead of standard tf.keras.layers.Rescaling (which had serialization issues),
        # implement a custom rescaling as a Lambda layer wrapping elementwise multiply.
        # This avoids config serialization issues with vector scale in newer Keras versions.
        scaling_vector = tf.convert_to_tensor(scaling_vector, dtype=tf.float32)
        # Reshape to match expected broadcasting dims: (1, 5)
        scaling_vector = tf.reshape(scaling_vector, (1, -1))
        self.rescaling = tf.keras.layers.Lambda(lambda x: x * scaling_vector, name="custom_rescaling")

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.rescaling(x)
        return x

def my_model_function():
    # Use the same scaling vector as in the examples to maintain consistency.
    # Here, an example vector is used from the typical "r" variable in the issue (random for illustration).
    # In real usage, this should be set to the actual scaling vector used during training or inference.
    example_scaling_vector = [0.5410176, 0.03500207, 0.687843, 0.8070028, 0.22955463]
    return MyModel(example_scaling_vector)

def GetInput():
    # Return a random tensor matching the model input: shape (batch, 10), dtype float32.
    # Batch size chosen is 1 for simplicity.
    return tf.random.uniform((1, 10), dtype=tf.float32)


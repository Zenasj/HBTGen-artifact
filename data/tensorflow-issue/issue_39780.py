# tf.random.uniform((1, 14), dtype=tf.float32) ← inferred input shape from input_signature

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the underlying keras model equivalent to umpire_regressor
        self.dense0 = tf.keras.layers.Dense(64, activation='relu', name="dense0")
        self.estimate = tf.keras.layers.Dense(1, activation='linear', name='estimate')
        # MSE loss instantiated once to avoid re-instantiating on each call
        self.mse = tf.keras.losses.MeanSquaredError()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(1, 14), dtype=tf.float32),  # 1d_features
        tf.TensorSpec(shape=(1, 1), dtype=tf.float32),   # true_action_values
    ])
    def fit_action(self, _1d_features, true_action_values):
        """
        Mimics the `fit_action` tf.function from the issue. Takes input features and true values,
        computes the estimated values from the network and the MSE loss.

        Note: The original bug was due to referencing a variable defined outside this scope
        causing serialization error. In this refactor, the logic and variables are scoped properly.

        Args:
            _1d_features: Tensor with shape (1, 14)
            true_action_values: Tensor with shape (1, 1)

        Returns:
            loss: scalar tensor representing mean squared error loss
        """
        x = self.dense0(_1d_features)
        estimated_action_values = self.estimate(x)
        loss = self.mse(true_action_values, estimated_action_values)
        return loss

    def call(self, inputs):
        # Standard forward pass for inference: input shape (1,14) -> output shape (1,1)
        x = self.dense0(inputs)
        return self.estimate(x)


def my_model_function():
    # Return an instance of MyModel with weights uninitialized (randomly initialized)
    return MyModel()

def GetInput():
    # Generate a single batch input tensor of shape (1, 14) matching input_signature
    return tf.random.uniform((1, 14), dtype=tf.float32)


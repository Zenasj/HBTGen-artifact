# tf.random.uniform((B, 2048), dtype=tf.float32) ← inferred input shape from Regressor call method: (batch_size, 2048)

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    """
    This model fuses the 'Regressor' model described in the issue,
    adapted to use fixed batch size logic to avoid TensorList ops in TFLite conversion.
    It implements the iterative prediction of 'theta' vectors using dense layers and tensor arrays,
    replaced with a fixed-size tf.Tensor to avoid dynamic shape ops like TensorListReserve/SetItem/Stack.

    It supports a fixed batch size set at initialization to eliminate dynamic TensorList ops.

    Forward output shape: (ITERATIONS, batch_size, 85)
    """

    def __init__(self, batch_size=1, iterations=3, name='my_model'):
        super(MyModel, self).__init__(name=name)

        # Config simulation (from issue references)
        self.BATCH_SIZE = batch_size
        self.ITERATIONS = iterations

        # mean_theta shape assumed (1, 85) per original (loaded externally in original)
        # Here we initialize trainable variable with zeros.
        self.mean_theta = tf.Variable(
            tf.zeros((1, 85), dtype=tf.float32), 
            name='mean_theta', 
            trainable=True)

        # Dense layers as in Regressor model
        self.fc_one = layers.Dense(1024, name='fc_0')
        self.dropout_one = layers.Dropout(0.5)
        self.fc_two = layers.Dense(1024, name='fc_1')
        self.dropout_two = layers.Dropout(0.5)

        variance_scaling = tf.initializers.VarianceScaling(
            scale=0.01, mode='fan_avg', distribution='uniform'
        )
        self.fc_out = layers.Dense(85, kernel_initializer=variance_scaling, name='fc_out')

    def call(self, inputs, training=False):
        """
        inputs: Tensor with shape (batch_size, 2048).
        Must have fixed batch size equal to self.BATCH_SIZE to avoid dynamic TensorList ops.

        Returns:
            Tensor of shape (ITERATIONS, batch_size, 85) with stacked thetas.
        """

        # Enforce fixed batch size from input shape or fallback to self.BATCH_SIZE
        batch_size = inputs.shape[0]
        if batch_size is None:
            batch_size = self.BATCH_SIZE

        # Check input feature dims match expected 2048 (safe-guard)
        if inputs.shape[1] != 2048:
            raise ValueError(f"Input shape mismatch, expected (_, 2048), got {inputs.shape}")

        # Expand mean_theta to batch size: [batch_size, 85]
        batch_theta = tf.tile(self.mean_theta, [batch_size, 1])

        # Prepare a tensor to hold thetas across iterations:
        # Instead of TensorArray (which uses TensorList ops), we use a fixed tensor buffer and tf.TensorArray with static size.
        # We'll accumulate results in a Python list and stack at the end to help TF compile statically.
        theta_list = []

        updated_theta = batch_theta
        for _ in range(self.ITERATIONS):
            # Concatenate inputs with current estimate theta
            total_inputs = tf.concat([inputs, updated_theta], axis=1)  # Shape: (batch_size, 2048+85=2133)

            # Pass through fc blocks: dense-relu-dropout pattern
            x = self.fc_one(total_inputs, training=training)
            x = tf.nn.relu(x)
            x = self.dropout_one(x, training=training)

            x = self.fc_two(x, training=training)
            x = tf.nn.relu(x)
            x = self.dropout_two(x, training=training)

            x = self.fc_out(x, training=training)  # Output shape: (batch_size, 85)

            # Update theta by adding the predicted offset
            updated_theta = updated_theta + x  # Shape: (batch_size, 85)

            theta_list.append(updated_theta)

        # Stack all theta outputs: shape (iterations, batch_size, 85)
        thetas_stacked = tf.stack(theta_list, axis=0)
        return thetas_stacked


def my_model_function():
    # Instantiate model with a default fixed batch size (e.g., 1) and number of iterations=3 as in original
    return MyModel(batch_size=1, iterations=3)


def GetInput():
    """
    Returns a valid input tensor matching the expected input shape for MyModel call:
    Shape: (batch_size, 2048)
    Dtype: tf.float32
    
    We pick batch_size = 1 by default consistent with model's batch_size.
    """
    batch_size = 1
    feature_dim = 2048
    # Random uniform inputs
    return tf.random.uniform((batch_size, feature_dim), dtype=tf.float32)


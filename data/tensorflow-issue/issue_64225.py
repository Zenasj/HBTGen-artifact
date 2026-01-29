# tf.random.uniform((32, 4, 4), dtype=tf.float32) ← Assumed batch size 32, square 4x4 matrices as example input shape

import tensorflow as tf
from tensorflow.keras.losses import Loss

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Example model to produce square matrices of shape (4,4)
        # Using kernel_initializer=tf.keras.initializers.Zeros() to simulate the reported bug context
        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dense2 = tf.keras.layers.Dense(16, activation='relu')
        # Output layer to produce 4*4=16 outputs, reshaped into (4,4) matrix
        # Kernel initialized to zeros to mirror the problematic scenario
        self.output_layer = tf.keras.layers.Dense(
            16,
            kernel_initializer=tf.keras.initializers.Zeros(),
            activation=None
        )
        self.reshape = tf.keras.layers.Reshape((4,4))

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.output_layer(x)
        x = self.reshape(x)
        return x

class DeterminantZeroLoss(Loss):
    def __init__(self, name="determinant_zero_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # Calculate the determinant of each predicted matrix in the batch.
        # y_pred shape is assumed (batch, 4, 4)
        # tf.linalg.det will return shape (batch,)
        # Penalize when determinant is NOT close to zero, because determinant=0 means matrix is singular.
        # According to discussion, low determinant should NOT be penalized.
        # So, a loss could be something that encourages determinant close to zero, e.g. abs(det).
        # But the original user says if matrix is NOT invertible (det=0), loss should be low.

        # Try-catch is ineffective in graph mode, so instead use tf.where to avoid invalid operations.

        # To mimic the behavior without triggering inversion, just use abs(det).
        # If input matrix is badly conditioned, tf.linalg.det may internally try matrix inversion (TF 2.16 bug).
        # Here we just return abs of determinant as loss, consistent with original intent.

        det = tf.linalg.det(y_pred)

        # Non-negative scalar loss: mean absolute determinant over batch
        loss = tf.reduce_mean(tf.abs(det))

        return loss

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input compatible with MyModel
    # Given the dense layers, input shape: (batch_size, features)
    # The first dense layer has 16 units, so input features can be any reasonable number.
    # Choosing input features=10 as example.
    batch_size = 32
    input_features = 10
    # Random uniform input
    inp = tf.random.uniform((batch_size, input_features), dtype=tf.float32)
    return inp


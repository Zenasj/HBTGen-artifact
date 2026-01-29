# tf.random.uniform((16, 1), dtype=tf.float32) ← input shape inferred from code: batch_size=16, feature_dim=1

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The constant left-hand matrix in the matmul operation as a tf.Variable or tf.constant
        # Shape (13,16) as per tf.matmul(tf.ones((13,16)), tf_input)
        # Using tf.ones initialization as in original code
        # This constant simulates the fixed left factor in the matmul
        self.left_const = tf.ones((13, 16), dtype=tf.float32)

    def call(self, inputs):
        # inputs shape: (batch=16, 1) is expected? Actually original input_shape = [16, 1]
        # But tf_input in original code: keras.Input(input_shape[1:], batch_size=input_shape[0])
        # input_shape[1:] = (1,), batch_size=16, so input tensor shape: (16, 1)
        # However, original code does tf.matmul(tf.ones((13,16)), tf_input)
        # tf.ones: (13,16)
        # tf_input: (16,1)
        # tf.matmul(13x16, 16x1) => 13x1 output

        # So the model expects input shape (batch=16, features=1)
        # Actually in keras.Input(input_shape[1:], batch_size=16), the shape argument is (1,)
        # But from tf.matmul, the input must be actually (16,1) matrix, so input shape (16,1)

        # But there's a conflict: keras.Input(input_shape[1:], batch_size=input_shape[0])
        # input_shape = [16,1]
        # input_shape[1:] = (1,), batch_size=16
        # So input tensor shape: (16,1)
        # The matmul does tf.matmul(tf.ones((13,16)), tf_input) → matmul(13x16, 16x1)

        # In call, inputs shape is (batch, features) = (16,1)

        # Actually tf.matmul expects last two dims, batch dims broadcasted
        # Here the left matrix is (13,16)
        # inputs = (16, 1)
        # This would cause error, so we need to explicitly control axes.

        # Given this, model assumes input tensor shape is (16,1)
        # The output of matmul(tf.ones(13,16), inputs) → (13,1)

        # So the batch dimension here is implicit in input shape or fixed to 16.

        # We'll implement it exactly as: output = matmul(left_const, inputs)
        # input shape: (16,1) tensor
        # output shape: (13,1)

        output = tf.matmul(self.left_const, inputs)
        return output

def my_model_function():
    # Return the instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor with shape (16, 1), dtype float32 matching expected input
    return tf.random.uniform((16, 1), dtype=tf.float32)


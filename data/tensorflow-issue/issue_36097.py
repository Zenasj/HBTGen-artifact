# tf.random.uniform((B, N), dtype=tf.float32) ← assuming input shape (batch, N) with N=2 based on provided usage example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Initialize weights for tridiagonal matrix: superdiagonal, main diagonal, subdiagonal
        # Shapes are (input_dim, 1). We'll delay setting input_dim till build is called.
        self.super_diag = None
        self.main_diag = None
        self.sub_diag = None

    def build(self, input_shape):
        # input_shape is (batch_size, N), N = input_shape[-1]
        input_dim = input_shape[-1]
        # Initialize weights as trainable variables for each diagonal
        self.super_diag = self.add_weight(
            shape=(input_dim, 1),
            initializer="random_normal",
            trainable=True,
            name="super_diagonal")
        self.main_diag = self.add_weight(
            shape=(input_dim, 1),
            initializer="random_normal",
            trainable=True,
            name="main_diagonal")
        self.sub_diag = self.add_weight(
            shape=(input_dim, 1),
            initializer="random_normal",
            trainable=True,
            name="sub_diagonal")

    def call(self, inputs):
        # inputs shape: (batch_size, N)
        # The tf.linalg.tridiagonal_matmul requires diagonals in (super, main, sub) order.
        # The inputs should be (..., N) or (..., N, K) tensor. Assuming K=1 here (since inputs are 2D).
        # If inputs has shape (batch, N), interpret as (batch, N, 1) for multiplication.
        # Reshape inputs to (batch, N, 1)
        inputs_expanded = tf.expand_dims(inputs, axis=-1)
        # Prepare diagonals tuple as required:
        # As per tf.linalg.tridiagonal_matmul: diagonals is tuple (super_diag, main_diag, sub_diag),
        # each shape (..., N), so we squeeze last dim from our weights.
        diagonals = (
            tf.squeeze(self.super_diag, axis=-1), 
            tf.squeeze(self.main_diag, axis=-1), 
            tf.squeeze(self.sub_diag, axis=-1)
        )
        # Multiply tridiagonal matrix by inputs
        result = tf.linalg.tridiagonal_matmul(diagonals, inputs_expanded, diagonals_format='sequence')
        # result shape: (batch, N, 1), squeeze last dim for output shape (batch, N)
        return tf.squeeze(result, axis=-1)

def my_model_function():
    # Instantiate and return MyModel instance
    return MyModel()

def GetInput():
    # Input shape should match (batch, input_dim)
    # The example used input shape (10, 2), so let's keep batch=10, N=2.
    batch_size = 10
    input_dim = 2
    # Generate random uniform input tensor in [0,1)
    return tf.random.uniform(shape=(batch_size, input_dim), dtype=tf.float32)


# tf.random.uniform((B, 5), dtype=tf.float32) ← Input shape (batch size, 5 features)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense_hidden = tf.keras.layers.Dense(24, activation='relu')
        self.dense_out = tf.keras.layers.Dense(1, name='main_output')
        # Lambda layer to produce second output as square of main output
        self.lambda_square = tf.keras.layers.Lambda(lambda x: x ** 2, name='second_output')

    def call(self, inputs, training=None):
        hidden = self.dense_hidden(inputs)
        out1 = self.dense_out(hidden)  # main output
        out2 = self.lambda_square(out1)  # second output
        return [out1, out2]

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Return a random input tensor of shape (batch_size, 5), dtype float32
    # Batch size chosen arbitrary as 16 to match usage in example
    return tf.random.uniform((16, 5), dtype=tf.float32)

# ---
# ### Explanation / Assumptions:
# - The input shape used here is `(batch_size, 5)` as inferred from the original example (`Input(shape=(5))`).
# - The model has two outputs:
#   - `main_output`: Dense(1) with ReLU activated hidden layer.
#   - `second_output`: Lambda layer that squares `main_output`.
# - The `call` method outputs both.
# - The `GetInput()` function returns random input tensor with batch size 16, consistent with the batching in original code.
# - No loss or metric logic is embedded here; this class just defines the model structure.
# - The naming and structure strictly follow the original model from the reported issue.
# - The model is compatible with TF 2.20.0 and exportable for XLA compilation.
# If you want, you could wrap usage like this:
# ```python
# model = my_model_function()
# @tf.function(jit_compile=True)
# def compiled(x):
#     return model(x)
# inputs = GetInput()
# outputs = compiled(inputs)
# ```
# This will run the model and allow XLA JIT compilation.
# Let me know if you want me to generate a version that incorporates metric computations or custom loss wrappers—as the issue was discussing—but the base model code requested here focuses on the core model.
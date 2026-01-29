# tf.random.uniform((B=1000, H=5), dtype=tf.float32) ← Input shape inferred from x: [1000, input_dim=5]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(100, activation=tf.nn.relu, kernel_initializer=None, use_bias=True)
        self.dense2 = tf.keras.layers.Dense(100, activation=tf.nn.relu, kernel_initializer=None, use_bias=True)
        self.dense3 = tf.keras.layers.Dense(1, activation=None, kernel_initializer=None, use_bias=True)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Build the model by providing an input shape; this triggers variable creation
    model.build((None, 5))
    return model

def GetInput():
    # Return a random input tensor matching (batch_size=1000, input_dim=5) as used in the original code
    # Use tf.random.normal to match np.random.normal behavior
    return tf.random.normal(shape=(1000, 5), mean=0.0, stddev=1.0, dtype=tf.float32)

# ---
# ### Explanation and assumptions:
# - Input shape `(1000, 5)` is inferred from the original training data arrays `x` with shape `[1000, input_dim]` where `input_dim = 5`.
# - The original model is a `Sequential` of three Dense layers with ReLU activations on the first two and no activation on the last. We replicate this exactly in the subclassed model.
# - Kernel initializer is `None` in the original snippet for simplicity; no special initialization specified — defaults apply.
# - `use_bias=True` because original code discussed bias causing non-determinism; it's kept as default.
# - We build the model once in `my_model_function` using the input shape, so weights are created before usage.
# - The `GetInput()` returns a tensor of shape `(1000, 5)` with normal distribution, matching the original input generation via `np.random.normal`.
# - We omit saving/loading or gradient computation logic, as the task is to produce the model and input generator.
# - This model and input tensor should work directly as `model(GetInput())` without errors.
# - This code is compatible with TensorFlow 2.20.0 and can be compiled with XLA using `@tf.function(jit_compile=True)`.
# - No test or main code included, per instructions.
# If you want, I can help add the gradient computation wrapped in a compiled function as well! Just let me know.
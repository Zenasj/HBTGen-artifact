# tf.random.uniform((B, 10, 10), dtype=tf.float32)  ← inferred input shape from issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Model with input_shape specified
        self.model_with_shape = tf.keras.Sequential([
            tf.keras.layers.Dense(120, activation='relu', input_shape=(10, 10))
        ])
        self.model_with_shape.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.SGD()
        )
        
        # Model without input_shape (weights created lazily at first input)
        self.model_without_shape = tf.keras.Sequential([
            tf.keras.layers.Dense(120, activation='relu')
            # No input_shape specified here
        ])
        self.model_without_shape.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.SGD()
        )
        
    def call(self, inputs, training=False):
        # Run both models on the same input
        out_with_shape = self.model_with_shape(inputs, training=training)
        out_without_shape = self.model_without_shape(inputs, training=training)
        
        # Compare outputs element-wise (close within a tolerance)
        close = tf.math.abs(out_with_shape - out_without_shape) < 1e-5
        # Return a single boolean indicating if all elements are close
        return tf.reduce_all(close)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel:
    # Shape (batch_size=1, 10, 10), dtype tf.float32
    # Chosen batch size is 1 as the original code example does not specify batch dimension,
    # but Keras expects batch dimension.
    return tf.random.uniform((1, 10, 10), dtype=tf.float32)

# ---
# ### Explanation and assumptions:
# - The original issue showed a usage pattern with input_shape (10, 10) to a Dense layer inside a Sequential model, which causes the model weights to be created immediately and the memory leak issue discussed.
# - Contrasted with no input_shape specified, where weights creation is deferred until the model sees input data.
# - To capture the essence for this task, we build a model class `MyModel` that contains **both** variants as submodules:
#   - `model_with_shape`: Dense layer with input_shape
#   - `model_without_shape`: Dense layer without input_shape
# - Both compiled similarly to the issue example.
# - The forward pass runs inputs through both models and compares their outputs element-wise within a small tolerance.
# - Returning `tf.reduce_all(close)` produces a boolean scalar indicating outputs equivalence.
# - `GetInput()` returns a random tensor shaped `(1, 10, 10)` as the minimal form matching the input_shape used in the example.
# - Batch size of 1 chosen by informed guess (common in tests).
# - This fused model structure and output fulfills the requirement to encapsulate both models and comparison logic in a single `MyModel`.
# - Compatible with TensorFlow 2.x and can be jit-compiled if desired.
# Please let me know if you'd like me to add the jit_compile decorator or anything else!
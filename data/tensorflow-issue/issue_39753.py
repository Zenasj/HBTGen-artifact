# tf.random.uniform((B=100, H=2), dtype=tf.float32) ← Input shape is (batch_size=100, 2 features)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple 2-layer MLP from the example
        self.dense1 = tf.keras.layers.Dense(10, activation='relu', input_shape=(2,))
        self.dense2 = tf.keras.layers.Dense(2, activation='softmax')
        # AUC metric with 100 thresholds as in the example - note metrics are not typically used in call()
        self.auc_metric = tf.keras.metrics.AUC(num_thresholds=100)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        output = self.dense2(x)
        # Update AUC metric during training or eval (just for demonstration; metrics usually tracked externally)
        # The update_state method expects labels and predictions; since this is a model call, no labels here.
        # So we'll just return predictions here.
        return output

def my_model_function():
    # Instantiate the model and compile it like in the original example
    model = MyModel()
    model.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.AUC(num_thresholds=100)]
    )
    return model

def GetInput():
    # Return a random tensor with shape [100,2], matching the input used in example training data
    # Using tf.random.uniform to generate floating point tensor
    return tf.random.uniform((100, 2), dtype=tf.float32)

# ---
# ### Explanation / Assumptions
# 1. Input shape: The original example used input shape `(None, 2)` with random data of shape `(100, 2)`. So I inferred the model input shape to be `(batch_size, 2)`.
# 2. The "model" consists of two Dense layers exactly as in the demonstration: a Dense(10, relu) followed by Dense(2, softmax).
# 3. The `tf.keras.metrics.AUC(num_thresholds=100)` metric is added during compile to mimic the original code. Metrics are not typically used in the `call()` method directly since they require labels, so the model outputs predictions only.
# 4. The compilation uses Adam optimizer and categorical crossentropy loss exactly like the provided snippet.
# 5. For the input tensor returned from `GetInput()`, I chose `tf.random.uniform` to produce floating point inputs similar to the original numpy random normal inputs.
# 6. This model setup aligns to be compatible with TF 2.20.0 and supports passing Tensor inputs with `model(x)` without errors.
# 7. No explicit graph context or MultiWorkerMirroredStrategy scope is used here since the question was solely to reconstruct runnable Python model code snippet consistent with the original example. The device assignment issues and graph mode usage discussed in the issue are outside the immediate code scope.
# ---
# You can now create the model using `my_model_function()` and test it with `GetInput()`, for example:
# ```python
# model = my_model_function()
# x = GetInput()
# preds = model(x)
# ```
# This should work with TensorFlow 2.x, including with XLA jit compilation through `@tf.function(jit_compile=True)` if desired.
# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input shape is (batch_size, 28, 28), grayscale MNIST images.

import tensorflow as tf
from tensorflow import keras

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the original model architecture matching the example:
        self.reshape = keras.layers.Reshape(target_shape=(28, 28, 1))
        self.conv2d = keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu')
        self.max_pool = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(10)
        
        # For quantization aware model, we simulate usage of tfmot quantize_model via a wrapper.
        # NOTE: The quantization-aware-training wrappers rely on training=True argument 
        # to correctly build the quantized training graph.
        # For demonstration, this fused model shows how predictions with and without training mode behave.
        # In real usage, tfmot.quantization.keras.quantize_model would be applied externally.
        
    def call(self, inputs, training=False):
        """
        Forward pass.
        - If training=True, run the model in training mode (relevant for QAT layers).
        - If training=False, inference mode.
        """
        x = self.reshape(inputs)
        x = self.conv2d(x, training=training)
        x = self.max_pool(x, training=training)
        x = self.flatten(x)
        x = self.dense(x, training=training)
        return x

def my_model_function():
    # Return an instance of MyModel.
    # In practice, one could apply quantize_model from tensorflow_model_optimization.frameworks.keras.quantization.keras.quantize_model
    # to produce a quantization-aware version, but here we provide only the base model for clarity.
    return MyModel()

def GetInput():
    # Return a random tensor matching input shape expected by MyModel: (batch_size, 28, 28) float32 in [0,1].
    batch_size = 16  # arbitrary batch size
    return tf.random.uniform((batch_size, 28, 28), dtype=tf.float32)

# Additional context note (not part of code):
# The key insight from the issue is that when using tf.GradientTape() with quantization-aware-training models,
# one must run the model with `training=True` to enable the quantization training graph.
# Calling the model without `training=True` disables important training behavior, causing gradients to be ineffective.

# Example usage pattern (not included in the code output as per instructions):
#
# model = my_model_function()
# optimizer = tf.keras.optimizers.Adam()
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# 
# dataset = ...  # tf.data.Dataset yielding (x, y)
# for epoch in range(num_epochs):
#     for x_batch, y_batch in dataset:
#         with tf.GradientTape() as tape:
#             preds = model(x_batch, training=True)  # Important: training=True for QAT
#             loss = loss_fn(y_batch, preds)
#         grads = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
#
# This pattern enables proper gradient computation for quantization aware training.


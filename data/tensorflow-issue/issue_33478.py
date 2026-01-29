# tf.random.uniform((B, 1), dtype=tf.float32)  # Assuming batch size B=4 for example input shape of (B, 1)

import tensorflow as tf
from typing import Callable, List, Optional

class MyModel(tf.keras.Model):
    """
    A tf.keras.Model subclass that supports registering hooks on layers to capture
    and optionally modify intermediate inputs and outputs during the forward pass.

    This design fuses the discussed ideas from the issue:
    - Wrapping layers or monkey patching layer.call to insert hooks
    - Hooks receive layer input and output tensors
    - Hooks can optionally replace output tensor
    - You can register multiple hooks per layer

    The forward pass applies the hooked layers sequentially.

    This design assumes the model is simple enough to apply hooks in this manner.
    """

    def __init__(self):
        super().__init__()
        # Define a simple model for illustration with two Dense layers
        self.dense1 = tf.keras.layers.Dense(50, input_shape=(1,), name="dense_1")
        self.relu = tf.keras.layers.ReLU(name="relu")
        self.dense2 = tf.keras.layers.Dense(1, name="dense_2")

        # Store hooks for each layer: Dict[layer] -> List[hook_fn]
        # A hook function has signature (input_tensor, output_tensor) -> Optional[output_tensor]
        self._hooks = {
            self.dense1: [],
            self.relu: [],
            self.dense2: [],
        }

        # Patch layers for hooks on their call methods
        for layer in self._hooks.keys():
            self._patch_layer_call(layer)

    def _patch_layer_call(self, layer: tf.keras.layers.Layer):
        # Save original call method
        original_call = layer.call

        def hooked_call(inputs, *args, **kwargs):
            # Call original method
            output = original_call(inputs, *args, **kwargs)
            # Apply hooks in order, each can modify output
            for hook in self._hooks[layer]:
                hook_result = hook(inputs, output)
                if hook_result is not None:
                    output = hook_result
            return output

        layer.call = hooked_call

    def register_hook(self, layer: tf.keras.layers.Layer,
                      hook_fn: Callable[[tf.Tensor, tf.Tensor], Optional[tf.Tensor]]):
        """
        Register a hook function on a specific layer.
        The hook function receives (input_tensor, output_tensor) and may return a new output tensor.
        """
        if layer not in self._hooks:
            raise ValueError("Layer not registered in this model for hooks.")
        self._hooks[layer].append(hook_fn)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # Sequential forward pass with hooked layers
        x = inputs
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Generate a random input tensor with batch size 4 and input dim 1 (as per model input)
    return tf.random.uniform((4, 1), dtype=tf.float32)


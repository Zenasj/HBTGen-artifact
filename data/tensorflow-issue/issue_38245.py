# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← inferred generic input shape for layers usually (batch, height, width, channels)

import tensorflow as tf
from typing import Callable, Optional, Dict, Any, Union

class MyModel(tf.keras.Model):
    """
    This class fuses the concept outlined for Serializer with Keras layers/optimizers registration,
    deserialization and retrieval to provide a unified API.

    It encapsulates layer and optimizer serializer submodules, mimicking the
    behavior described in the GitHub issue for reducing code duplication and improving
    extensibility in tf.keras.[layers, optimizers].

    The forward call here just performs a simple pass-thru to the internal sublayers for demo
    because the original issue was around serialization, not a forward model.
    """

    class Serializer(object):
        def __init__(self,
                     accepted_types: Optional[type] = None,
                     module_name: Optional[str] = None):
            self._module_objects: Dict[str, Callable] = {}
            self._module_name = module_name
            self._accepted_types = accepted_types

        def register(self, fn_or_class: Callable):
            assert callable(fn_or_class), "Must register a callable"
            name = fn_or_class.__name__
            if name in self._module_objects:
                raise ValueError(
                    f'Could not register {name} - already in {self._module_name} module_objects')
            self._module_objects[name] = fn_or_class
            return fn_or_class

        def deserialize(self, config: Union[str, dict, Callable], custom_objects=None):
            return tf.keras.utils.deserialize_keras_object(
                config,
                module_objects=self._module_objects,
                custom_objects=custom_objects,
                printable_module_name=self._module_name)

        def _get_fallback(self, identifier):
            raise ValueError(
                f'Could not interpret {self._module_name} identifier: {identifier}')

        def get(self, identifier: Union[str, dict, Callable]):
            if isinstance(identifier, self._accepted_types):
                return identifier
            elif isinstance(identifier, str):
                identifier = dict(class_name=identifier, config={})
            if isinstance(identifier, dict):
                return self.deserialize(identifier)
            else:
                return self._get_fallback(identifier)

    class OptimizerSerializer(Serializer):
        def __init__(self):
            # Use tf.keras.optimizers.Optimizer as accepted type and module name 'optimizers'
            super().__init__(tf.keras.optimizers.Optimizer, 'optimizers')

        def _get_fallback(self, identifier):
            # Fallback: if already a tf.keras.optimizers.Optimizer instance,
            # wrap or track it accordingly (mimicking TFOptimizer and K.track_tf_optimizer)
            if isinstance(identifier, tf.keras.optimizers.Optimizer):
                # Provide a dummy pass-through: in real TF, wrapping occurs here
                # For placeholder, just return the identifier itself
                return identifier
            else:
                return super()._get_fallback(identifier)

    def __init__(self):
        super().__init__()
        # Instantiate serializers for layers and optimizers
        self.layers_serializer = self.Serializer(tf.keras.layers.Layer, 'layers')
        self.optimizers_serializer = self.OptimizerSerializer()

        # For demonstration - register a custom layer internally
        @self.layers_serializer.register
        class MyLayer(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.dense = tf.keras.layers.Dense(4)

            def call(self, inputs):
                return self.dense(inputs)

        # Internal example optimizer registration could be done via optimizers_serializer.register
        # but typically optimizers are already known.
        # We'll just keep the serializers separate.

        # Use a simple internal layer to demonstrate a forward pass:
        self.example_layer = self.layers_serializer._module_objects['MyLayer']()

    @tf.function
    def call(self, inputs, training=None):
        # Simple forward using the registered example layer
        return self.example_layer(inputs)

    # Convenience methods to expose serializers' interface like `get`, `register`, `deserialize`

    def layers_get(self, identifier):
        return self.layers_serializer.get(identifier)

    def layers_register(self, fn_or_class):
        return self.layers_serializer.register(fn_or_class)

    def layers_deserialize(self, config, custom_objects=None):
        return self.layers_serializer.deserialize(config, custom_objects)

    def optimizers_get(self, identifier):
        return self.optimizers_serializer.get(identifier)

    def optimizers_register(self, fn_or_class):
        return self.optimizers_serializer.register(fn_or_class)

    def optimizers_deserialize(self, config, custom_objects=None):
        return self.optimizers_serializer.deserialize(config, custom_objects)


def my_model_function():
    """
    Return an instance of MyModel with serializers initialized.
    """
    return MyModel()


def GetInput():
    """
    Return a valid random input tensor matching the input shape that MyModel expects.

    Assumptions:
    - Input to the internal Dense layer inside MyLayer likely expects at least 2D,
      so we use a 2D tensor: (batch_size, features)
    - Using batch size 1 and input dim 8 for demonstration.
    """
    batch_size = 1
    input_dim = 8
    return tf.random.uniform((batch_size, input_dim), dtype=tf.float32)


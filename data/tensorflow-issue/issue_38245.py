from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf

register = tf.keras.utils.register_keras_serializable('my_package.layers')

@register
class MyLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

layer = MyLayer()
config = tf.keras.utils.serialize_keras_object(layer)

print(tf.keras.layers.deserialize(config))      # works - but why no layers.get?
print(tf.keras.optimizers.get(config))          # works - but should it?

class Serializer(object):
    def __init__(self, accepted_types=None, module_name=None):
        self._module_objects = {}
        self._module_name = module_name
        self._accepted_types = accepted_types
    
    def register(self, fn_or_class: Callable):
        assert callable(fn_or_class)
        name = fn_or_class.__name__
        if name in self._module_objects:
            raise ValueError(
                f'Could register {name} - already in {self._module_name} '
                'module_objects')
        self._module_objects[name] = fn_or_class
        return fn_or_class
    
    def deserialize(self, config, custom_objects=None):
        return tf.keras.utils.deserialize_keras_object(
            config, module_objects=self._module_objects,
            custom_objects=custom_objects,
            printable_module_name=self._module_name)
    
    def _get_fallback(self, identifier):
        raise ValueError(
            f'Could not interpret optimizer identifier: {identifier}')
    
    def get(self, identifier):
        if isinstance(identifier, self._accepted_types):
            return identifier
        elif isinstance(identifier, str):
            identifier = dict(class_name=identifier, config={})
        if isinstance(identifier, dict):
            return self.deserialize(identifier)
        else:
            return self._get_fallback(identifier)

# in layers.py
_serializer = Serializer(Layer, 'layers')
get = _serializer.get
deserialize = _serializer.deserialize
register = _serializer.register

# in optimizers.py
class OptimizerSerializer(Serializer):

    def __init__(self):
        self.__init__(Optimizer, 'optimizers')

    def _get_fallback(self, identifier):
        if isinstance(identifier, tf_optimizer_module.Optimizer):
            opt = TFOptimizer(identifier)
            K.track_tf_optimizer(opt)
            return opt
        else:
            return super()._get_fallback(identifier)

_serializer = OptimizerSerializer()
get = _serializer.get
deserialize = _serializer.deserialize
register = _serializer.register
# tf.random.uniform((B, H, W, C), dtype=...) ← No specific input shape given in the issue; input shape irrelevant here as this is a Layer subclass fix for variable tracking
import itertools
from functools import wraps
from typing import Any, Callable, Optional, Sequence

import tensorflow as tf


def extend_and_filter(
    extend_method: Callable[..., Sequence], filter_method: Optional[Callable[..., Sequence]] = None,
) -> Callable[[Any], Any]:
    """
    This decorator calls a decorated method, and extends the result with another method
    on the same class. This method is called after the decorated function, with the same
    arguments as the decorated function. If specified, a second filter method can be applied
    to the extended list. Filter method should also be a method from the class.

    :param extend_method: Callable
        Accepts the same argument as the decorated method.
        The returned list from `extend_method` will be added to the
        decorated method's returned list.
    :param filter_method: Callable
        Takes in the extended list and filters it.
        Defaults to no filtering for `filter_method` equal to `None`.
    """

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapped(self, *args, **kwargs):  # type: ignore
            ret = f(self, *args, **kwargs)
            ret.extend(extend_method(self, *args, **kwargs))
            ret = filter_method(self, ret) if filter_method is not None else ret
            return ret

        return wrapped

    return decorator


class MyModel(tf.keras.layers.Layer):
    """
    A tf.keras Layer subclass that properly tracks variables on sub-attributes
    that are generic tf.Modules (not only sublayers). This resolves the
    common issue where Keras layers only track variables from sub-Layers,
    but not from arbitrary tf.Modules.

    This implementation extends trainable_weights, non_trainable_weights,
    trainable_variables, and variables properties by also including variables
    that belong to any tf.Module found as attributes (or inside lists/tuples/dicts).
    """

    @property
    def _submodules(self) -> Sequence[tf.Module]:
        """
        Return a list of tf.Module instances that are attributes on the class.
        This includes direct attributes that are tf.Modules, and also 
        nested modules within lists, tuples, or dicts of attributes.

        Duplicates are removed while preserving order.
        """
        submodules = []

        def get_nested_submodules(*objs: Any) -> None:
            # Helper to recursively find tf.Module instances nested in containers
            for o in objs:
                if isinstance(o, tf.Module):
                    submodules.append(o)

        for key, obj in self.__dict__.items():
            if isinstance(obj, tf.Module):
                submodules.append(obj)
            elif isinstance(obj, (list, tuple)):
                tf.nest.map_structure(get_nested_submodules, obj)
            elif isinstance(obj, dict):
                tf.nest.map_structure(get_nested_submodules, obj.values())

        # Remove duplicates and maintain order (Python 3.6+ dict preserves insertion order)
        return list(dict.fromkeys(submodules))

    def submodule_variables(self) -> Sequence[tf.Variable]:
        """
        Return flat iterable of variables from the attributes that are tf.Modules.
        """
        return list(itertools.chain(*[module.variables for module in self._submodules]))

    def submodule_trainable_variables(self) -> Sequence[tf.Variable]:
        """
        Return flat iterable of trainable variables from attributes that are tf.Modules.
        """
        return list(itertools.chain(*[module.trainable_variables for module in self._submodules]))

    def submodule_non_trainable_variables(self) -> Sequence[tf.Variable]:
        """
        Return flat iterable of non-trainable variables from attributes that are tf.Modules.
        """
        # gather all variables from submodules and filter those that are not trainable
        return [v for module in self._submodules for v in module.variables if not v.trainable]

    def _dedup_weights(self, weights):  # type: ignore
        """
        Deduplicate weights while maintaining order as much as possible.
        Copied from the superclass to use locally.
        """
        return super()._dedup_weights(weights)

    @property  # type: ignore
    @extend_and_filter(submodule_trainable_variables, _dedup_weights)
    def trainable_weights(self) -> Sequence[tf.Variable]:
        return super().trainable_weights

    @property  # type: ignore
    @extend_and_filter(submodule_non_trainable_variables, _dedup_weights)
    def non_trainable_weights(self) -> Sequence[tf.Variable]:
        return super().non_trainable_weights

    @property  # type: ignore
    @extend_and_filter(submodule_trainable_variables, _dedup_weights)
    def trainable_variables(self) -> Sequence[tf.Variable]:
        return super().trainable_variables

    @property  # type: ignore
    @extend_and_filter(submodule_variables, _dedup_weights)
    def variables(self) -> Sequence[tf.Variable]:
        return super().variables


def my_model_function():
    """
    Return an instance of the Trackable Layer (MyModel).
    """
    return MyModel()


def GetInput():
    """
    Return a dummy tensor input for running MyModel.

    Since MyModel is just a Layer subclass intended to fix variable tracking,
    and no specific call() method or input shape is defined in the issue,
    assume a generic input shape of (1, 1) float32 tensor.

    Users can adapt this accordingly for their use case.
    """
    return tf.random.uniform((1, 1), dtype=tf.float32)


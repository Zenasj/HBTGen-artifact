import torch.nn as nn

class Module:
    def _forward_unimplemented(self, *input: Any) -> None:
        raise NotImplementedError

    r"""Defines the computation performed at every call.

    Should be overridden by all subclasses.

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        registered hooks while the latter silently ignores them.
    """
    forward: Callable[..., Any] = _forward_unimplemented

class implements:  # pylint: disable=invalid-name
    """Mark a function as implementing an interface."""

    def __init__(self, interface: Type):
        """Instantiate the decorator.

        Args:
            interface: the interface that is implemented
        """
        self.interface = interface

    def __call__(self, func: _F) -> _F:
        """Take a function and return it unchanged."""
        super_method = getattr(self.interface, func.__name__, None)
        assert super_method is not None, f"'{func.__name__}' does not exist in {self.interface}"
        assert super_method.__doc__, f"'{super_method}' has no docstring"
        return func

import torch
from torch import nn
assert getattr(nn.Module, nn.Module.forward.__name__, None).__doc__

# Trick mypy into not applying contravariance rules to inputs by defining
# forward as a value, rather than a function.  See also
# https://github.com/python/mypy/issues/8795
def _forward_unimplemented(self, *input: Any) -> None:
    raise NotImplementedError

def _forward_unimplemented(self, *input: Any) -> None:
    '''Trick mypy into not applying contravariance rules to inputs by defining
    forward as a value, rather than a function.  See also
    https://github.com/python/mypy/issues/8795
    '''
    raise NotImplementedError
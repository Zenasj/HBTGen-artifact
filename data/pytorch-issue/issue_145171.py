import torch

from typing import Any, Callable, Optional, TypeVar
from typing_extensions import ParamSpec, TypeVarTuple, Unpack

import hashlib
import pickle
import inspect

from torch._prims.context import TorchRefsMode
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx, wrapper_and_args_for_make_fx

T = TypeVar("T")
P = ParamSpec("P")
Ts = TypeVarTuple("Ts")


def execute(
    gm: GraphModule,
    *args: Unpack[Ts],
    executor: str = "aten",
    executor_parameters: Optional[dict] = None,
) -> Any:
    """
    Prototype ATen executor.

    Just executes the context's graph.
    """
    if executor == "aten":
        return gm.forward(*args)

    msg = f"Received unexpected value for 'executor': {executor}. Allowed values are: aten."
    raise ValueError(msg)


def compute_cache_key(fn: Callable, args: tuple, kwargs: dict) -> str:
    """
    Compute a unique key for the function and its parameters (args, kwargs).
    The key is based on the function's source code and serialized arguments.
    """
    fn_code = pickle.dumps(inspect.getsource(fn).encode("utf-8"))
    args_data = pickle.dumps((args, kwargs))
    return hashlib.sha256(fn_code + args_data).hexdigest()


_cache = {}


def make_traced(fn: Callable[P, T]) -> Callable[P, T]:
    """
    Returns a tracked function that uses caching for reuse
    the graphs already drawn previously.
    """
    def _traced(*args: P.args, **kwargs: P.kwargs) -> T:
        executor = str(kwargs.pop("executor", "aten"))

        cache_key = compute_cache_key(fn, args, kwargs)

        if cache_key in _cache:
            gm = _cache[cache_key]
        else:
            wrapped, all_args = wrapper_and_args_for_make_fx(fn, args, kwargs)

            with TorchRefsMode():
                gm = make_fx(wrapped)(all_args)

            _cache[cache_key] = gm

        return execute(gm, *args, executor=executor)

    return _traced
import numpy as np

def is_seq(obj):
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes))

__import__("some_module.some_member")
net.register_buffer("buf", buf)

__import__(b"some_module.some_member")
net.register_buffer(b"buf", buf)

class MyPythonStore(dist.Store):
    def __init__(self):
        super().__init__()
        self.store = {}

    def set(self, key, value):
        if not isinstance(key, (str, bytes)):  # <=== reverted
            raise AssertionError("Expected set to be called with string key")
        if type(value) is not bytes:
            raise AssertionError("Expected set to be called with bytes value")
        self.store[key] = value
    def get(self, key):
        value = self.store.get(key, b"")
        if type(value) is not bytes:
            raise AssertionError("Expected get to return bytes value")
        return value
    def add(self, key, value):
        new = int(self.store.get(key, 0)) + value
        self.set(key, bytes(str(new).encode("utf-8")))
        return new

def test_doc(self):
        checked_types = (types.MethodType, types.FunctionType,
                         types.BuiltinFunctionType, types.BuiltinMethodType)
        def _test_namespace(ns, *skips):
            if isinstance(ns, object):
                ns_name = ns.__class__.__name__
            else:
                ns_name = ns.__name__
            skip_regexes = []
            for r in skips:
                if isinstance(r, str):  # <=== leave as `str`-only
                    skip_regexes.append(re.compile('^{}$'.format(re.escape(r))))
                else:
                    skip_regexes.append(r)

            ...

def set_default_tensor_type(t):
    r"""Sets the default ``torch.Tensor`` type to floating point tensor type
    ``t``. This type will also be used as default floating point type for
    type inference in :func:`torch.tensor`.
    The default floating point tensor type is initially ``torch.FloatTensor``.
    Args:
        t (type or string): the floating point tensor type or its name
    Example::
        >>> # xdoctest: +SKIP("Other tests may have changed the default type. Can we reset it?")
        >>> torch.tensor([1.2, 3]).dtype    # initial default for floating point is torch.float32
        torch.float32
        >>> torch.set_default_tensor_type(torch.DoubleTensor)
        >>> torch.tensor([1.2, 3]).dtype    # a new floating point tensor
        torch.float64
    """
    if isinstance(t, str):  # <=== leave as `str`-only
        t = _import_dotted_name(t)
    _C._set_default_tensor_type(t)

# Casts Tensors and containers of Tensors.  Special-cases passthroughs for strings and np.ndarrays, which
# may be falsely detected as "Iterables."
def _cast(value, dtype):
    if isinstance(value, torch.Tensor):
        is_eligible = (value.is_floating_point() and value.is_cuda and (value.dtype is not torch.float64))
        return value.to(dtype) if is_eligible else value
    elif isinstance(value, (str, bytes)):  # <=== reverted
        return value
    elif HAS_NUMPY and isinstance(value, np.ndarray):
        return value
    elif isinstance(value, collections.abc.Mapping):
        return {_cast(k, dtype): _cast(v, dtype) for k, v in value.items()}
    elif isinstance(value, collections.abc.Iterable):
        iterable = map(lambda v: _cast(v, dtype), value)
        if isinstance(value, (list, tuple)):
            return type(value)(iterable)
        else:
            return iterable
    else:
        return value

def __new__(cls, name: str):
        if not isinstance(name, str):  # <=== leave as `str`-only
            raise ValueError("Backend name must be a string, but got: {}".format(name))
        value = getattr(Backend, name.upper(), Backend.UNDEFINED)

        if value != Backend.GLOO and value != Backend.NCCL and value != Backend.UCC and value != Backend.MPI:
            value = name.lower()
        return value

def rendezvous(url: str, rank: int = -1, world_size: int = -1, **kwargs):
    if not isinstance(url, str):  # <=== leave as `str`-only
        raise RuntimeError("`url` must be a string. {}: {}".format(type(url), url))

    if not isinstance(rank, numbers.Integral):
        raise RuntimeError("`rank` must be an integer. {}".format(rank))

    ...

def load(f, map_location=None, _extra_files=None, _restore_shapes=False):
    r"""
    Load a :class:`ScriptModule` or :class:`ScriptFunction` previously
    saved with :func:`torch.jit.save <torch.jit.save>`
    All previously saved modules, no matter their device, are first loaded onto CPU,
    and then are moved to the devices they were saved from. If this fails (e.g.
    because the run time system doesn't have certain devices), an exception is
    raised.
    Args:
        f: a file-like object (has to implement read, readline, tell, and seek),
            or a string containing a file name
        map_location (string or torch.device): A simplified version of
            ``map_location`` in `torch.jit.save` used to dynamically remap
            storages to an alternative set of devices.
        _extra_files (dictionary of filename to content): The extra
            filenames given in the map would be loaded and their content
            would be stored in the provided map.
        _restore_shapes (bool): Whether or not to retrace the module on load using stored inputs
    Returns:
        A :class:`ScriptModule` object.
    Example:
    .. testcode::
        import torch
        import io
        torch.jit.load('scriptmodule.pt')
        # Load ScriptModule from io.BytesIO object
        with open('scriptmodule.pt', 'rb') as f:
            buffer = io.BytesIO(f.read())
        # Load all tensors to the original device
        torch.jit.load(buffer)
        # Load all tensors onto CPU, using a device
        buffer.seek(0)
        torch.jit.load(buffer, map_location=torch.device('cpu'))
        # Load all tensors onto CPU, using a string
        buffer.seek(0)
        torch.jit.load(buffer, map_location='cpu')
        # Load with extra files.
        extra_files = {'foo.txt': ''}  # values will be replaced with data
        torch.jit.load('scriptmodule.pt', _extra_files=extra_files)
        print(extra_files['foo.txt'])
    .. testoutput::
        :hide:
        ...
    .. testcleanup::
        import os
        os.remove("scriptmodule.pt")
    """

    if isinstance(f, str):  # <=== leave as `str`-only
        if not os.path.exists(f):  # type: ignore[type-var]
            raise ValueError("The provided filename {} does not exist".format(f))  # type: ignore[str-bytes-safe]
        if os.path.isdir(f):
            raise ValueError("The provided filename {} is a directory".format(f))  # type: ignore[str-bytes-safe]
    map_location = validate_map_location(map_location)
    if _extra_files is None:
        _extra_files = {}
    
    ...

@_beartype.beartype
def _is_onnx_list(value):
    return (
        not isinstance(value, (str, bytes))  # <=== reverted
        and not isinstance(value, torch.Tensor)
        and isinstance(value, Iterable)
    )

def _get_restore_location(map_location):
    if map_location is None:
        restore_location = default_restore_location
    elif isinstance(map_location, dict):
        def restore_location(storage, location):
            location = map_location.get(location, location)
            return default_restore_location(storage, location)
    elif isinstance(map_location, str):  # <=== leave as `str`-only
        def restore_location(storage, location):
            return default_restore_location(storage, map_location)
    elif isinstance(map_location, torch.device):
        def restore_location(storage, location):
            return default_restore_location(storage, str(map_location))
    else:
        def restore_location(storage, location):
            result = map_location(storage, location)
            if result is None:
                result = default_restore_location(storage, location)
            return result
    return restore_location

def shell(command, cwd=None, env=None, stdout=None, stderr=None):
    sys.stdout.flush()
    sys.stderr.flush()
    # The following cool snippet is copied from Py3 core library subprocess.call
    # only the with
    #   1. `except KeyboardInterrupt` block added for SIGINT handling.
    #   2. In Py2, subprocess.Popen doesn't return a context manager, so we do
    #      `p.wait()` in a `final` block for the code to be portable.
    #
    # https://github.com/python/cpython/blob/71b6c1af727fbe13525fb734568057d78cea33f3/Lib/subprocess.py#L309-L323
    assert not isinstance(command, str), "Command to shell should be a list or tuple of tokens"  # <=== leave as `str`-only
    p = subprocess.Popen(command, universal_newlines=True, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
    return wait_for_process(p)

class StringPair(UnittestPair):
    CLS = (str, bytes)  # <=== reverted
    TYPE_NAME = "string"

def default_convert(data):
    r"""
        Function that converts each NumPy array element into a :class:`torch.Tensor`. If the input is a `Sequence`,
        `Collection`, or `Mapping`, it tries to convert each element inside to a :class:`torch.Tensor`.
        If the input is not an NumPy array, it is left unchanged.
        This is used as the default function for collation when both `batch_sampler` and
        `batch_size` are NOT defined in :class:`~torch.utils.data.DataLoader`.
        The general input type to output type mapping is similar to that
        of :func:`~torch.utils.data.default_collate`. See the description there for more details.
        Args:
            data: a single data point to be converted
        Examples:
            >>> # xdoctest: +SKIP
            >>> # Example with `int`
            >>> default_convert(0)
            0
            >>> # Example with NumPy array
            >>> default_convert(np.array([0, 1]))
            tensor([0, 1])
            >>> # Example with NamedTuple
            >>> Point = namedtuple('Point', ['x', 'y'])
            >>> default_convert(Point(0, 0))
            Point(x=0, y=0)
            >>> default_convert(Point(np.array(0), np.array(0)))
            Point(x=tensor(0), y=tensor(0))
            >>> # Example with List
            >>> default_convert([np.array([0, 1]), np.array([2, 3])])
            [tensor([0, 1]), tensor([2, 3])]
    """
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data)
    elif isinstance(data, collections.abc.Mapping):
        try:
            return elem_type({key: default_convert(data[key]) for key in data})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, tuple):
        return [default_convert(d) for d in data]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):  # <=== reverted
        try:
            return elem_type([default_convert(d) for d in data])
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return [default_convert(d) for d in data]
    else:
        return data

default_collate_fn_map[str] = collate_str_fn
default_collate_fn_map[bytes] = collate_str_fn  # <=== reverted

def pin_memory(data, device=None):
    if isinstance(data, torch.Tensor):
        return data.pin_memory(device)
    elif isinstance(data, (str, bytes)):  # <=== reverted
        return data
    elif isinstance(data, collections.abc.Mapping):
        try:
            return type(data)({k: pin_memory(sample, device) for k, sample in data.items()})  # type: ignore[call-arg]
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {k: pin_memory(sample, device) for k, sample in data.items()}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return type(data)(*(pin_memory(sample, device) for sample in data))
    elif isinstance(data, tuple):
        return [pin_memory(sample, device) for sample in data]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence):
        try:
            return type(data)([pin_memory(sample, device) for sample in data])  # type: ignore[call-arg]
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return [pin_memory(sample, device) for sample in data]
    elif hasattr(data, "pin_memory"):
        return data.pin_memory()
    else:
        return data
import torch

py
class DP(IterableDataset["invalid_type"]):
    def __iter__(self) -> "invalid_type":
        ...

class DP(IterableDataset[str]):
    def __iter__(self) -> str:
        ...

class DP(IterableDataset[str]):
    def __iter__(self) -> Iterator[int]:
        ...

py
# 1
class DS(IterableDataset["invalid_type"]):  # Raise Exception from the evaluation `eval("invalid_type", globals, locals)`
    def __iter__(self) -> "invalid_type":
        ...

# 2
class DS(IterableDataset[str]):
    def __iter__(self) -> str:  # Raise `TypeError: Expected 'Iterator' as the return annotation for '__iter__' of DS, but found str`
        ...

# 3
class DS(IterableDataset[str]):
    def __iter__(self) -> Iterator[int]:  # Raise `TypeError: Expected return type of '__iter__' is a subtype of str, but found int`
        ...

# 4
class DS(IterableDataset[str], metaclass=MyMeta):  # Raise `TypeError: metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass of the metaclasses of all its bases`
    pass

py
class DP:
    def __init__(self, dp: DataFrameDataPipe["name": str, "val": float]):
        ...

py
class GenericNamedTuple(Generic[T_co], NamedTuple):  # TypeError: metaclass conflict
    name: str
    data: T_co

py
class GenericNamedTuple(Generic[T_co], NamedTuple):
    name: str
    data: T_co
class DP(IterableDataset["GenericNamedTuple[torch.Tensor]"]):  # Use string
   ...

py
class GenericNamedTuple(Generic[T_co], NamedTuple):  # TypeError: metaclass conflict
    name: str
    data: T_co

py
class GenericNamedTuple(Generic[T_co], NamedTuple):
    name: str
    data: T_co
class DP(IterableDataset["GenericNamedTuple[torch.Tensor]"]):  # TypeError: 'type' object is not subscriptable
   ...
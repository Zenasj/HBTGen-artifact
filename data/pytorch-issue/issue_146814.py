import torch

class ClassAMock:
    class Nested:
        pass

class ClassBMock:
    class Nested:
        pass

def test_nested_class() -> None:
    torch.save(
        dict(
            a_nested=ClassAMock.Nested(),
            b_nested=ClassBMock.Nested(),
        ),
        'nested_class.pth'
    )
    torch.serialization.add_safe_globals(
        [ClassAMock, ClassBMock, getattr, ClassAMock.Nested, ClassBMock.Nested]
    )
    torch.load('nested_class.pth')

test_nested_class()

import torch
torch.__version__

class ClassAMock:
    class Nested:
        pass

class ClassBMock:
    class Nested:
        pass

def test_nested_class() -> None:
    torch.save(
        dict(
            a_nested=ClassAMock.Nested(),
            b_nested=ClassBMock.Nested(),
        ),
        'nested_class.pth'
    )
    torch.serialization.add_safe_globals(
        [ClassAMock, ClassBMock, getattr, (ClassAMock.Nested, "__main__.ClassAMock.Nested"), (ClassBMock.Nested, "__main__.ClassBMock.Nested")]
    )
    torch.load('nested_class.pth')

test_nested_class()
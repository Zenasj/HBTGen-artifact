import pytest

class Foo:
    def __repr__(self):
        return str(id(self))

@pytest.mark.parametrize(
    "bar",
    [
        pytest.param(type),
        pytest.param(lambda obj: obj.__class__),
        pytest.param(Foo()),
    ],
)
def test_foo(bar):
    pass
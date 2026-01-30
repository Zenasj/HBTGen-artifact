import torch

from typing import Dict, Any


class Test:
    def foo(self) -> Dict[str, Any]:
        result = {
            "int": 123,
            "float": 0.123,
            "str": "abc",
        }
        return result


class Test2:
    def foo(self) -> Dict[str, Any]:
        result = {
            "int": 123,
            "float": 0.123,
            "str": "abc",
        }
        assert isinstance(result, Dict[str, Any])
        return result


if __name__ == "__main__":
    t = Test()
    t2 = Test2()

    # Will trigger the following error:
    # Return value was annotated as having type Dict[str, Any] but is actually of type Dict[str, Union[float, int, str]]:
    try:
        t_script = torch.jit.script(t)
    except Exception as e:
        print(e)

    # This one is okay
    t2_script = torch.jit.script(t2)

    # However calling foo will trigger the following error:
    # RuntimeError: AssertionError:
    # Notice that t2.foo() will also trigger error:
    # TypeError: Subscripted generics cannot be used with class and instance checks
    try:
        t2_script.foo()
    except Exception as e:
        print(e)

result: Dict[str, Any] = {
            "int": 123,
            "float": 0.123,
            "str": "abc",
        }
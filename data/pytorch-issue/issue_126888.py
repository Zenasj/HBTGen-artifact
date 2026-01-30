from typing_extensions import deprecated


@deprecated(
    "Function `deprecated_func` is deprecated and will be removed in a future release. Use `new_func` instead."
)
def deprecated_func():
    return "deprecated_func"


def new_func():
    return "new_func"


@deprecated(
    "Class `DeprecatedClass` is deprecated and will be removed in a future release. Use `NewClass` instead."
)
class DeprecatedClass:
    pass


class InheritedDeprecatedClass(DeprecatedClass):
    pass


class NewClass:
    pass


class Foo:
    @deprecated("Method `Foo.bar` is deprecated and will be removed in a future release.")
    def bar(self):
        print("Hello, world!")


if __name__ == "__main__":
    print(deprecated_func())
    print(new_func())
    instance = DeprecatedClass()
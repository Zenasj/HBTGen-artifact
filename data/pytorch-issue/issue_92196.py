import torch

from torch import package
from torch._dynamo.testing import make_test_cls_with_patches

try:
    from . import (
        test_misc,
    )
except ImportError:
    import test_misc

import unittest


def make_dynamic_cls(cls):
    return make_test_cls_with_patches(
        cls, "DynamicShapes", "_dynamic_shapes", ("dynamic_shapes", True)
    )


DynamicShapesMiscTests = make_dynamic_cls(test_misc.MiscTests)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    # Having an imported module here will trigger the error
    path = "/tmp/MyPickledModule.pt"
    package_name = "MyPickledModule"
    resource_name = "MyPickledModule.pkl"

    imp = package.PackageImporter(path)
    loaded_model = imp.load_pickle(package_name, resource_name)

    run_tests()
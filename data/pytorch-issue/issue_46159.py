import torch
import torch.nn as nn

class MyTensor(torch.Tensor):
    ...
ten = MyTensor(10)
assert isinstance(ten, MyTensor)
assert isinstance(ten + 1, MyTensor), "I expect this to fail"

class MyTensor(torch.Tensor):
    ...
ten = MyTensor(10)
assert isinstance(ten, MyTensor)
assert not isinstance(ten + 1, MyTensor), "all good"

class Distance2PoincareHyperplanes(torch.nn.Module):
    n = 0
    # 1D, 2D versions of this class ara available with a one line change
    # class Distance2PoincareHyperplanes2d(Distance2PoincareHyperplanes):
    #     n = 2

    def __init__(
        self,
        plane_shape: int,
        num_planes: int,
        signed=True,
        squared=False,
        *,
        ball,
        std=1.0,
    ):
        super().__init__()
        self.signed = signed
        self.squared = squared
        # Do not forget to save Manifold instance to the Module
        self.ball = ball
        self.plane_shape = geoopt.utils.size2shape(plane_shape)
        self.num_planes = num_planes

        # In a layer we create Manifold Parameters in the same way we do it for
        # regular pytorch Parameters, there is no difference. But geoopt optimizer
        # will recognize the manifold and adjust to it
        self.points = geoopt.ManifoldParameter(
            torch.empty(num_planes, plane_shape), manifold=self.ball
        )
        self.std = std
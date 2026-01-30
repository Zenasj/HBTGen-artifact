import torch

from torch.distributions import Distribution, constraints

class MyDist(Distribution):

    arg_constraints = {
        "optional_arg1": constraints.greater_than(0),
        "optional_arg2": constraints.greater_than(0),
    }
    support = constraints.positive

    def __init__(
        self,
        optional_arg1: float | None = None, 
        optional_arg2: float | None = None,
        validate_args: bool = False,
    ):
        self.optional_arg1 = optional_arg1
        self.optional_arg2 = optional_arg2
        super().__init__(validate_args=validate_args)


print(MyDist(optional_arg1=1.0))
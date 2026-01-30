import torch.nn as nn
from functorch import make_functional, jvp

class LinearizedModel(nn.Module):
    """Creates a linearized version of a nn.Module.

    The linearized version of a model is a proper PyTorch model and can be
    trained as any other nn.Module.

    Args:
        model (nn.Module): The model to linearize.
    """

    def __init__(self, model: nn.Module) -> None:
        """Initializes the linearized model."""
        super().__init__()
        func0, params0 = make_functional(model.eval(), disable_autograd_tracking=True)

        # We wrap the func0 module in a lambda function to hide its "meta"
        # parameters from the parameter list.
        self.func0 = lambda params, x: func0(params, x)

        self.params = nn.ParameterList([nn.Parameter(p.clone()) for p in params0])
        self.params0 = nn.ParameterList(params0)

        # We don't want to train the initialization parameters.
        for p in self.params0:
            p.requires_grad = False

    def __call__(self, x) -> torch.Tensor:
        """Computes the linearized model output using a first-order Taylor decomposition."""
        dparams = [p - p0 for p, p0 in zip(self.params, self.params0)]
        _, dp = jvp(
            lambda param: self.func0(param, x),
            (tuple(self.params0),),
            (tuple(dparams),),
        )
        return self.func0(self.params0, x) + dp

import torch
import torch.nn as nn
from functorch import make_functional, jvp

class LinearizedModel(nn.Module):
  ...

model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
linearized_model = LinearizedModel(model)

devices = [0, 1]
parallel_model = nn.DataParallel(linearized_model, device_ids=devices)
parallel_model = parallel_model.cuda()

x = torch.randn(10, 2).cuda()
y = parallel_model(x)
import torch.nn as nn

import math
import pyro
import torch
import pyro.nn as pnn
import torch.nn as tnn
import pyro.distributions as dist
from torch import Tensor
from typing import Iterable
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal


class BNN(pnn.PyroModule):
    def __init__(
        self,
        input_size: int,
        hidden_layer_sizes: Iterable[int],
        output_size: int,
    ) -> None:
        super().__init__()

        layer_sizes = (
            [(input_size, hidden_layer_sizes[0])]
            + list(zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:]))
            + [(hidden_layer_sizes[-1], output_size)]
        )

        layers = [
            pnn.PyroModule[tnn.Linear](in_size, out_size)
            for in_size, out_size in layer_sizes
        ]
        self.layers = pnn.PyroModule[tnn.ModuleList](layers)

        # make the layers Bayesian
        for layer_idx, layer in enumerate(self.layers):
            layer.weight = pnn.PyroSample(
                dist.Normal(0.0, 5.0 * math.sqrt(2 / layer_sizes[layer_idx][0]))
                .expand(
                    [
                        layer_sizes[layer_idx][1],
                        layer_sizes[layer_idx][0],
                    ]
                )
                .to_event(2)
            )
            layer.bias = pnn.PyroSample(
                dist.Normal(0.0, 5.0).expand([layer_sizes[layer_idx][1]]).to_event(1)
            )

        self.activation = tnn.Tanh()
        self.output_size = output_size

    def forward(self, x: Tensor, obs=None) -> Tensor:
        mean = self.layers[-1](x)

        if obs is not None:
            with pyro.plate("data", x.shape[0]):
                pyro.sample(
                    "obs", dist.Normal(mean, 0.1).to_event(self.output_size), obs=obs
                )

        return mean


class FailingBNN(BNN):
    def __init__(
        self, input_size: int, hidden_layer_sizes: Iterable[int], output_size: int
    ) -> None:
        super().__init__(input_size, hidden_layer_sizes, output_size)

    def forward(self, x: Tensor, obs=None) -> Tensor:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)

        return super().forward(x, obs=obs)


class WorkaroundBNN(BNN):
    def __init__(
        self, input_size: int, hidden_layer_sizes: Iterable[int], output_size: int
    ) -> None:
        super().__init__(input_size, hidden_layer_sizes, output_size)

    def forward(self, x: Tensor, obs=None) -> Tensor:
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)

        return super().forward(x, obs=obs)


class NestedBNN(pnn.PyroModule):
    def __init__(self, bnns: Iterable[BNN]) -> None:
        super().__init__()
        self.bnns = pnn.PyroModule[tnn.ModuleList](bnns)

    def forward(self, x: Tensor, obs=None) -> Tensor:
        # mean = torch.mean(torch.stack([bnn(x) for bnn in self.bnns]), dim=0)
        mean = sum([bnn(x) for bnn in self.bnns]) / len(self.bnns)

        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Normal(mean, 0.1).to_event(1), obs=obs)

        return mean


def train_bnn(model: BNN, input_size: int) -> None:
    pyro.clear_param_store()

    # small numbers for demo purposes
    num_points = 20
    num_svi_iterations = 100

    x = torch.linspace(0, 1, num_points).reshape((-1, input_size))
    y = torch.sin(2 * math.pi * x) + torch.randn(x.size()) * 0.1

    guide = AutoDiagonalNormal(model)
    adam = pyro.optim.Adam({"lr": 0.03})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    for _ in range(num_svi_iterations):
        svi.step(x, y)


def run() -> None:
    # small numbers for demo purposes
    input_size = 1
    output_size = 1
    hidden_size = 3
    num_hidden_layers = 3

    train_bnn(
        FailingBNN(input_size, [hidden_size] * num_hidden_layers, output_size),
        input_size,
    )
    print("Successfully trained FailingBNN")

    train_bnn(
        WorkaroundBNN(input_size, [hidden_size] * num_hidden_layers, output_size),
        input_size,
    )
    print("Successfully trained WorkaroundBNN")

    train_bnn(
        NestedBNN(
            [
                WorkaroundBNN(
                    input_size, [hidden_size] * num_hidden_layers, output_size
                ),
                WorkaroundBNN(
                    input_size, [hidden_size] * num_hidden_layers, output_size
                ),
            ]
        ),
        input_size,
    )
    print("Successfully trained NestedBNN with WorkaroundBNNs")

    train_bnn(
        NestedBNN(
            [
                FailingBNN(input_size, [hidden_size] * num_hidden_layers, output_size),
                FailingBNN(input_size, [hidden_size] * num_hidden_layers, output_size),
            ]
        ),
        input_size,
    )
    print("Successfully trained NestedBNN with FailingBNNs")


if __name__ == "__main__":
    run()

# Remove the last element in my module list
self.my_mod = self.my_mod[:-1]
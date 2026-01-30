import torch.nn as nn

import os
import tempfile

import torch


def main():
    """Main."""
    model = torch.nn.Linear(3, 5)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, max_lr=1e-5, total_steps=123
    )

    _, f = tempfile.mkstemp()
    try:
        torch.save(lr_scheduler.state_dict(), f)
        # Evidently, when saving (and therefore, reloading) the LRS's state, the "anneal_func"
        # is serialized as a bound method of the optimizer. This means it has a reference, via
        # `__self__`, to the optimizer instance. This in turn means that the _entire_ optimizer
        # object itself is pickled into the state - which means so is its reference to the
        # optimizer!
        lr_state = torch.load(f)

        print(lr_state["anneal_func"].__self__.optimizer.param_groups)
    finally:
        os.remove(f)


if __name__ == "__main__":
    main()
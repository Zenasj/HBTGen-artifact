import torch

targets = [single_targets, [x * 10 for x in single_targets]]

from torch.testing._internal.common_utils import parametrize, instantiate_parametrized_tests

@parametrize("T_mult", [1, 2, 4])
def test_CosineAnnealingWarmRestarts_lr1(self, T_mult):
    ...

...

# At the bottom of the file before `if __name__ == '__main__':`
instantiate_parametrized_tests(TestLRScheduler)
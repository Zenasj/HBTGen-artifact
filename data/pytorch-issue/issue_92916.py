import torch
import functorch
def test():
    self_tensor = torch.tensor([1, 2, 3, 4])
    found_inf = torch.tensor(0)
    inv_scale = torch.tensor(0.2)
    print(torch._amp_foreach_non_finite_check_and_unscale_([self_tensor], found_inf, inv_scale))

functorch.functionalize(test)()
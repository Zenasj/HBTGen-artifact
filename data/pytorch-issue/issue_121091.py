import torch
import torch.nn as nn

def sanity_check_spectral_norm(dtype):
    """
    Check that the spectral norm is indeed correctly computed
    for a simple transformation diag(2., 1.). (spectral norm = 2.)
    """
    net = nn.Linear(2, 2).to(dtype)
    with torch.no_grad():
        # top singular value is 2 = spectral norm
        id = torch.diag(torch.tensor([2., 1.], dtype=dtype))
        net.weight = nn.Parameter(id)
        # weights should be rescaled by spectral norm
        torch.nn.utils.parametrizations.spectral_norm(net, n_power_iterations=400)
        # weights should now be diag(1., 0.5)
        assert torch.allclose(net.weight, torch.diag(torch.tensor([1., 0.5])))


# works for floats
sanity_check_spectral_norm(torch.float)

# doesn't work for complex-valued floats
sanity_check_spectral_norm(torch.cfloat)
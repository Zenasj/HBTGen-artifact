import torch

from torch.utils.tensorboard import SummaryWriter

from packaging.version import Version

if not hasattr(tensorboard, "__version__") or Version(
    tensorboard.__version__
) < Version("1.15"):
    raise ImportError("TensorBoard logging requires TensorBoard version 1.15 or above")

del Version
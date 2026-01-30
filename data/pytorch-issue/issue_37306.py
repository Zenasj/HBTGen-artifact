import torch
import torchvision

# copied from torchvision.extension.py
lib_dir = os.path.dirname(__file__)
loader_details = (
    importlib.machinery.ExtensionFileLoader,
    importlib.machinery.EXTENSION_SUFFIXES
)

extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
torch.ops.load_library(extfinder.find_spec("_C").origin)

a = torch.rand(3, 3, requires_grad=True)
b = torch.rand(3, 3, requires_grad=True)
out = torch.ops.my_ops.add(a, b)
print(out)
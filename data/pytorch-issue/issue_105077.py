import torch

from torch._subclasses import fake_tensor
import transformers

fake_mode = fake_tensor.FakeTensorMode()
with fake_mode:
    fake_model = transformers.AutoModel.from_pretrained("sshleifer/tiny-gpt2")  # raises OSError: Unable to load weights from pytorch checkpoint file for '...' at  If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True.

class PytorchPatcher:
    def __init__(self):

        def torch__util__rebuild_tensor_wrapper(storage, storage_offset, size, stride):
            from torch._subclasses.fake_tensor import FakeTensorMode
            from torch.utils._mode_utils import no_dispatch
            from torch.utils._python_dispatch import _get_current_dispatch_mode

            def _rebuild_real_tensor(storage, storage_offset, size, stride):
                t = torch.tensor(
                    [], dtype=storage.dtype, device=storage._untyped_storage.device
                )
                return t.set_(storage._untyped_storage, storage_offset, size, stride)

            mode = _get_current_dispatch_mode()
            if isinstance(mode, FakeTensorMode):
                # Create a real tensor and then convert it to FakeTensor.
                # We cannot directly create a FakeTensor because it tensor.set_(...)
                # is not supported in FakeTensorMode dispatcher.
                with no_dispatch():
                    t = _rebuild_real_tensor(storage, storage_offset, size, stride)
                return mode.from_tensor(t)
            return _rebuild_real_tensor(storage, storage_offset, size, stride)

        # Original version of torch.load.
        self.torch__util_rebuild_tensor = torch._utils._rebuild_tensor

        # Wrapper or modified version of torch functions.
        self.torch__util_rebuild_tensor_wrapper = torch__util__rebuild_tensor_wrapper

    def __enter__(self):
        torch._utils._rebuild_tensor = self.torch__util_rebuild_tensor_wrapper

    def __exit__(self, exc_type, exc_value, traceback):
        torch._utils._rebuild_tensor = self.torch__util_rebuild_tensor
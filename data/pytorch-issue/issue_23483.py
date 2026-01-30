import torch

class TracerWarning(Warning):

    @staticmethod
    def ignore_lib_warnings():

        # We ignore warnings from all submodules excluding the JIT, because we need them e.g. for _check_trace
        warnings.filterwarnings('ignore', category=TracerWarning, module='torch.(?!jit)')
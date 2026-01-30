import torch

# Represents a rule indicating how to xfail a particular test. It allows granularity
# at the device, dtype, op, and individual sample levels. This flexibility allows entire
# bugs to be represented by a single rule, even if this corresponds with multiple conceptual
# test cases across multiple ops.
@dataclass
class XFailRule:
    # expected error type
    error_type: TypeVar = Exception
    # expected error message
    error_msg: str = ".*"
    # function to indicate whether the rule applies; return True if so
    match_fn: Callable[[torch.device, torch.dtype, OpInfo, SampleInput], bool] = None
    # optional name for identifying the rule
    name: str = ""

    def match(self, device, dtype, op, sample) -> bool:
        return self.match_fn(device, dtype, op, sample)
import torch

@dataclass
class SampleRule(ABC):
    # function to indicate whether the rule applies to this op; return True if so
    # NB: str arg of callable is device_type
    op_match_fn: Callable[[str, OpInfo], bool] = None
    # function to indicate whether the rule applies to this sample; return True if so
    sample_match_fn: Callable[[torch.device, SampleInput], bool] = None
    # optional name for identifying the rule
    name: str = ""

@dataclass
class XFailRule(SampleRule):
    # expected error type
    error_type: TypeVar = Exception
    # expected error message
    error_msg: str = ".*"

@dataclass
class SkipRule(SampleRule):
    ...

class MyTestCase(TestCase):
    @ops(op_db)
    def test_foo(self, device, dtype, op):
        for sample in op.sample_inputs(device, dtype, requires_grad=False):
            # do some SampleInput-based test logic
            output = op.op(sample.input, *sample.args, **sample.kwargs)
            ...

# NB: Define rules for per-SampleInput xfails / skips. These can also be defined in-line in the @ops decorator, but
# it can be more readable to maintain these somewhere else. These are attempted to be matched in order and 
# the first one that matches applies, so order can matter.
FOO_SKIPS_AND_XFAILS = [
    XFailRule(
        error_type=ValueError,
        error_mg="2D inputs not supported",
        op_match_fn=lambda device, op: (
            # NB: logic for which ops this rule applies to goes here
            op.full_name == "add"
        ),
        sample_match_fn=lambda device, sample: (
            # NB: logic which samples this rule applies to goes here
            sample.input.dim() == 2
        ),
        # NB: optional rule identifier can help with debugging matched rules
        name="add_with_2D_inputs_not_supported",
    ),
    # NB: This follows a similar structure as XFailRule but without error_type / error_msg. Obviously
    # this skips a particular SampleInput instead of xfailing :)
    SkipRule(...),
    ...
]

class MyTestCase(TestCase):
    @ops(op_db)
    @sample_skips_and_xfails(FOO_SKIPS_AND_XFAILS)
    # NB: the @ops decorator automatically filters out any rules that don't apply to this op
    def test_foo(self, device, dtype, op):
        for sample, subtest_ctx in op.sample_inputs(
            # NB: use_subtests=True is required for skips / xfails to work. If skips / xfails are defined and use_subtests != True,
            # an informative error will be thrown.
            device, dtype, requires_grad=False, use_subtests=True
        ):
            # NB: this subtest context manager runs each sample input as a "subtest" and handles skips / xfails appropriately
            with subtest_ctx(self):
                # do some SampleInput-based test logic
                output = op.op(sample.input, *sample.args, **sample.kwargs)
                ...

...
# pre-existing xfail
xfail("item"),
# new granular xfail using syntactic sugar over the general system
xfailIf(
    "to",
    lambda sample: (
        sample.kwargs["memory_format"] == torch.channels_last
    ),
),
...
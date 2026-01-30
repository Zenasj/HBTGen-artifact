import pytest
import torch


def test_pass():
    def fn(x):
        return x + x

    input = torch.tensor(2.0)
    input2 = torch.tensor(-2.0)
    torch.compile(fn, backend="inductor", options={"_raise_error_for_testing": False})(input)  # this passes as it should
    torch.compile(fn, backend="inductor", options={"_raise_error_for_testing": True})(input2)  # this line should fail yet it doesn't
    assert True


def test_fail():
    def fn(x):
        return x + x

    input = torch.tensor(-2.0)
    torch.compile(fn, backend="inductor", options={"_raise_error_for_testing": True})(input) # this fails as expected
    assert True
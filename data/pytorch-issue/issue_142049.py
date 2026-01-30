import torch
import torch.nn as nn

def  test_cross_entropy_loss():
    """
    cross_entropy_loss": [
        "def cross_entropy_loss({}) -> Tensor: ...".
        format(
            ", ".join(
                [
                    "input: Tensor",
                    "target: Tensor",
                    "weight: Optional[Tensor] = None",
                    "reduction: int = 1",
                    "ignore_index: int = -100",
                    "label_smoothing: float = 0.0",
                ]
            )
        )
    """
    try:
        torch._C._nn.cross_entropy_loss(InvalidType(), InvalidType())
        raise Exception # these make sure the above code raises an error
    except Exception as e:
        assert e.__str__() == "cross_entropy_loss(): argument 'input' (position 1) must be Tensor, not InvalidType", e.__str__()

    try:
        torch._C._nn.cross_entropy_loss(t, InvalidType())
        raise Exception
    except Exception as e:
        assert e.__str__() == "cross_entropy_loss(): argument 'target' (position 2) must be Tensor, not InvalidType", e.__str__()

    try:
        torch._C._nn.cross_entropy_loss(t, t, InvalidType())
        raise Exception
    except Exception as e:
        assert e.__str__() == "cross_entropy_loss(): argument 'weight' (position 3) must be Tensor, not InvalidType", e.__str__()

    try:
        torch._C._nn.cross_entropy_loss(t, t, t, InvalidType())
        raise Exception
    except Exception as e:
        assert e.__str__() == "cross_entropy_loss(): argument 'reduction' (position 4) must be int, not InvalidType", e.__str__()

    try:
        torch._C._nn.cross_entropy_loss(t, t, t, i, InvalidType())
        raise Exception
    except Exception as e:
        assert e.__str__() == "cross_entropy_loss(): argument 'ignore_index' (position 5) must be int, not InvalidType", e.__str__()

    try:
        torch._C._nn.cross_entropy_loss(t, t, t, i, i, InvalidType())
        raise Exception
    except Exception as e:
        assert e.__str__() == "cross_entropy_loss(): argument 'label_smoothing' (position 6) must be float, not InvalidType", e.__str__()

    try:
        torch._C._nn.cross_entropy_loss(t, t, t, i, i, f)
        raise Exception
    except Exception as e:
        assert e.__str__() == "ignore_index is not supported for floating point target", e.__str__() # means it ran the code

    try:
        torch._C._nn.cross_entropy_loss(t, t, t, i, i, f, out=InvalidType())
        raise Exception
    except Exception as e:
        assert e.__str__() == "cross_entropy_loss() got an unexpected keyword argument 'out'", e.__str__()
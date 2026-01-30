import torch

xfail(
        "index_put",
        matcher=lambda sample: (sample.args[0][0].dtype == torch.bool) and (sample.kwargs.get("accumulate") == False),
        reason=onnx_test_common.reason_dynamo_does_not_support("Unknown reason!"),
    ),
import torch
import torch._inductor.metrics
from torch._dynamo import config as dynamo_config


@dynamo_config.patch({"capture_scalar_outputs": True})
def test_vertical_pointwise_reduction_fusion():
    # reset in case we run both cpu and cuda tests
    torch._inductor.metrics.reset()

    # Tests fusing a pointwise & reduction op with unbacked numel/rnumel.
    def fn(x, y, repeats):
        u0 = repeats.item()
        unbacked = y.expand(u0, *y.shape)  # [u0, 1, 16]
        # Note: We add x to both pointwise and reduction. Otherwise, the
        # scheduler will refuse to fuse ops whose only common buffer has
        # unbacked symints.
        pointwise = unbacked + x
        reduction = torch.sum(pointwise + x)
        return pointwise, reduction

    example_inputs = (
        torch.randn(32, 16).cuda(),
        torch.randn(1, 16).cuda(),
        torch.tensor(32).cuda(),
    )
    _ = torch.compile(fn, fullgraph=True)(*example_inputs)
    assert torch._inductor.metrics.generated_kernel_count == 1, f"expect kernel count 1, got {torch._inductor.metrics.generated_kernel_count}"

test_vertical_pointwise_reduction_fusion()
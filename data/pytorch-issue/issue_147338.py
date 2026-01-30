import torch
import torch.nn as nn

def test_torch_decomposition_keep_metadata() -> None:
    """Make sure the metadata is kept after exported program run_decompositions."""

    @torch.library.custom_op("mylib::add", mutates_args=())
    def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...

    @torch.library.register_fake("mylib::add")
    def _(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    class TestModel(torch.nn.Module):
        def forward(self, x, y):
            return torch.ops.mylib.add(x, y)

    model = TestModel()
    x_example = torch.randn(2, 3)
    y_example = torch.randn(2, 3)
    exported_program = torch.export.export(model, (x_example, y_example))

    for node in exported_program.graph.nodes:
        node.meta["my_field"] = "dummy"
    for node in exported_program.graph.nodes:
        assert node.meta["my_field"] == "dummy"

    decomposed_program = exported_program.run_decompositions()
    for node in decomposed_program.graph.nodes:
        assert node.meta["my_field"] == "dummy"  # This errors out because custom metadata is lost
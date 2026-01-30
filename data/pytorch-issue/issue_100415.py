import torch

def test_export_preserve_constraints_as_metadata(self):
        from torch._export.constraints import constrain_as_size, constrain_as_value

        def f(x, y):
            b = x.item()
            constrain_as_size(b, min=2, max=5)  # 1st inline constraint
            c = y.dim()
            constrain_as_value(c, min=1, max=3)  # 2nd inline constraint
            z = y[0:c]
            return torch.empty((b, y.shape[0])), z

        x = torch.tensor([3])
        y = torch.randn([8, 8, 6])
        example_inputs = [x, y]
        constraints = [dynamic_dim(y, 0) >= 6, dynamic_dim(y, 0) <= 10]
        gm, _ = torch._dynamo.export(
            f,
            *example_inputs,
            constraints=constraints,
            aten_graph=True,
            tracing_mode="symbolic",
        )
        print(gm.print_readable())
        print("DEBUG: inline_constraints")
        for k, vr in gm.meta["inline_constraints"].items():
            print(f"DEBUG  {k}:{vr}")
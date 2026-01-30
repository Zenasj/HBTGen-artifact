import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyModel(torch.nn.Module):

    def __init__(self):
        super(DummyModel, self).__init__()

        # Basic set of filters
        self.conv_down1 = nn.Conv2d(1, 16, 3, 1, 0)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_up1 = nn.Conv2d(16, 1, 3, 1, 1)

    def forward(self, x):
        x = self.conv_down1(x)
        x = self.activation(x)

        # Upsample
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # Onnx export fails here
        x = self.conv_up1(x)
        x = self.activation(x)

        return x


if __name__ == "__main__":
    # Declare the dummy model and confirm it works
    dummy_model = DummyModel()
    dummy_model = dummy_model.cuda()
    dummy_model.eval()

    fake_input = torch.randn(1, 1, 100, 100).cuda()
    fake_output = dummy_model(fake_input)

    # Now try to export it to onnx with dynamic sizing
    export_options = torch.onnx.ExportOptions(
        dynamic_shapes=True,  # Export works when set to False
    )  # Allow dynamic sizing
    onnx_program = torch.onnx.dynamo_export(
        dummy_model, fake_input, export_options=export_options
    )
    onnx_program.save("dummy_model.onnx")

import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Basic set of filters
        self.conv_down1 = nn.Conv2d(1, 16, 3, 1, 0)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_up1 = nn.Conv2d(16, 1, 3, 1, 1)

    def forward(self, x):
        x = self.conv_down1(x)
        x = self.activation(x)

        # Upsample
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # Onnx export fails here
        x = self.conv_up1(x)
        x = self.activation(x)

        return x


if __name__ == "__main__":
    # Declare the dummy model and confirm it works
    dummy_model = DummyModel()
    dummy_model.eval()

    fake_input = torch.randn(1, 1, 100, 100)
    fake_output = dummy_model(fake_input)

    ep = torch.export.export(
        dummy_model,
        (fake_input,),
        dynamic_shapes=[
            {
                2: torch.export.Dim("height", min=64),
                3: torch.export.Dim("width", min=64),
            },
        ],
    )

    print(ep)

    onnx_program = torch.onnx.export(ep, dynamo=True, report=True)

    print(onnx_program)
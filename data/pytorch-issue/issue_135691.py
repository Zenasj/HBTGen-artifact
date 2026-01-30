import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np

export_shape = (249, 1, 1000)
test_shape = (150, 1, 1000)


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.hidden_size = 500
        self.num_heads = 4
        self.attention = nn.MultiheadAttention(
            embed_dim=2 * self.hidden_size, num_heads=self.num_heads
        )

    def forward(self, input_x):
        return self.attention(input_x, input_x, input_x)


def export_onnx():
    input_x = torch.randn(export_shape)

    model = MyModule()
    model.eval()

    torch.onnx.export(
        model,
        input_x,
        "test.onnx",
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={
            "x": {0: "batch"},
            "y": {0: "batch"},
        },
    )


def infer_by_onnx(shape):
    input_x = torch.randn(shape)

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    ort_sess = ort.InferenceSession(
        "./test.onnx",
        sess_options=sess_options,
        providers=ort.get_available_providers(),
    )

    y = ort_sess.run(["y"], {"x": input_x.numpy()})[0]


def test_onnx():
    infer_by_onnx(export_shape)
    infer_by_onnx(test_shape)


if __name__ == "__main__":
    export_onnx()
    test_onnx()
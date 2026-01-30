import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn as nn
import onnx
import onnxruntime.backend as backend


class LstmModel(torch.nn.Module):
    def __init__(self):
        super(LstmModel, self).__init__()
        self.rnn = nn.LSTM(4, 3, batch_first=True)

    def forward(self, input_seq, length):
        packed_emb = pack_padded_sequence(input_seq, length, batch_first=True)
        rnn_outputs = self.rnn(packed_emb)
        hidden_states, _ = pad_packed_sequence(rnn_outputs[0], batch_first=True)
        return hidden_states


if __name__ == "__main__":
    # 1.test example_input with batch_size = 2
    # B * T * V = 2 * 2 * 4
    example_input = torch.tensor(
        [[[1.0, 2.0, 3.0, 0.0], [1.0, 2.0, 3.0, 0.0]], [[1.0, 2.0, 3.0, 0.0], [1.0, 2.0, 3.0, 0.0]]], dtype=torch.float)
    example_length = torch.tensor([2, 2])

    # 2.test example_input with batch_size = 1
    # B * T * V = 1 * 4 * 4
    example_input = torch.tensor(
        [[[1.0, 2.0, 3.0, 0.0], [1.0, 2.0, 3.0, 0.0], [1.0, 2.0, 3.0, 0.0], [1.0, 2.0, 3.0, 0.0]]], dtype=torch.float)
    example_length = torch.tensor([2])

    # export model by tracing
    model = LstmModel()
    with torch.no_grad():
        trace_out = model(example_input, example_length)
    input_names = ['input_seq', 'length']
    output_names = ['out']
    dynamic_axes = {'input_seq': {0: "batch_size", 1: "var_len"},
                    'length': {0: "batch_size"}}
    dummy_input = (example_input, example_length)
    torch.onnx.export(model, dummy_input, 'trace_model.onnx',
                      opset_version=13, example_outputs=trace_out, verbose=True,
                      input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)

    # load the onnx model to inference
    trace_model_onnx = onnx.load('trace_model.onnx')

    # have a inference
    print("change tethe batch_size to 3")
    mock_input = torch.tensor(
        [[[1.0, 2.0, 3.0, 0.0], [1.0, 2.0, 3.0, 0.0], [1.0, 2.0, 3.0, 0.0]],
         [[1.0, 2.0, 3.0, 0.0], [1.0, 2.0, 3.0, 0.0], [1.0, 2.0, 3.0, 0.0]],
         [[1.0, 2.0, 3.0, 0.0], [1.0, 2.0, 3.0, 0.0], [1.0, 2.0, 3.0, 0.0]]], dtype=torch.float)
    mock_length = torch.tensor([2, 2, 2])

    y_onnx = backend.run(
        trace_model_onnx,
        [mock_input.numpy(), mock_length.numpy()],
        device='CPU'
    )
    print(y_onnx)